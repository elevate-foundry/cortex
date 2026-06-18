"""
Gossip Transport — HTTP client/server for distributed peer state sync.

Bridges the SCL gossip protocol (src/scl/gossip.py) with the Cortex daemon's
HTTP interface. Peers exchange Braille fingerprints and deltas over HTTP:

  POST /v1/gossip          — receive a GossipMessage from a peer
  POST /v1/gossip/peers    — register a new peer
  GET  /v1/gossip/peers    — list known peers
  GET  /v1/gossip/state    — get our current state fingerprint
  GET  /v1/gossip/stats    — gossip statistics

Background task (_gossip_task) periodically initiates sync rounds with peers.
"""

import json
import logging
import random
import time
import asyncio
from dataclasses import asdict
from typing import Optional

import aiohttp

from .scl.gossip import GossipMessage, GossipMessageType, Peer, Swarm, MergeStrategy
from .scl.delta import Delta, SemanticState, VectorClock
from .braille.fingerprint import fingerprint, similarity
from .memory import Memory

logger = logging.getLogger("cortex.gossip")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GOSSIP_INTERVAL_SECONDS = 30     # How often to try gossiping with peers
GOSSIP_TIMEOUT_SECONDS = 10      # HTTP timeout per request
GOSSIP_PEERS_KEY = "gossip_peers"  # Memory policy key for peer list


# ---------------------------------------------------------------------------
# GossipTransport
# ---------------------------------------------------------------------------

class GossipTransport:
    """
    Manages network communication for gossip between Cortex nodes.

    Each node maintains a local Peer (with SemanticState + DeltaStream)
    and periodically initiates sync with remote peers over HTTP.
    """

    def __init__(
        self,
        memory: Memory,
        node_id: Optional[str] = None,
        listen_host: str = "127.0.0.1",
        listen_port: int = 11411,
    ):
        self.memory = memory
        self.node_id = node_id or f"cortex-{listen_port}"
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.base_url = f"http://{listen_host}:{listen_port}"

        # Local peer state
        self.local_peer = Peer(
            agent_id=self.node_id,
            initial_state=SemanticState(),
            merge_strategy=MergeStrategy.PRIORITY,
            weight=1.0,
        )

        # Known remote peers: {peer_id: "http://host:port"}
        self.remote_peers: dict[str, str] = {}
        self._load_peers_from_memory()

        # Track last sync time per peer
        self._last_sync: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Peer management
    # ------------------------------------------------------------------

    def add_peer(self, peer_id: str, url: str) -> None:
        """Register a remote peer."""
        if peer_id == self.node_id:
            logger.warning("Cannot add self as peer")
            return
        self.remote_peers[peer_id] = url.rstrip("/")
        self._save_peers_to_memory()
        logger.info("Added gossip peer: %s @ %s", peer_id, url)

    def remove_peer(self, peer_id: str) -> bool:
        """Remove a remote peer."""
        if peer_id in self.remote_peers:
            del self.remote_peers[peer_id]
            self._save_peers_to_memory()
            logger.info("Removed gossip peer: %s", peer_id)
            return True
        return False

    def list_peers(self) -> list[dict]:
        """List all known peers with their status."""
        now = time.monotonic()
        return [
            {
                "id": peer_id,
                "url": url,
                "last_sync_seconds_ago": round(now - self._last_sync.get(peer_id, 0), 1),
                "local": False,
            }
            for peer_id, url in self.remote_peers.items()
        ]

    def _load_peers_from_memory(self) -> None:
        """Load persisted peer list from SQLite."""
        peers_json = self.memory.get_policy(GOSSIP_PEERS_KEY, default="{}")
        if isinstance(peers_json, str):
            try:
                peers_json = json.loads(peers_json)
            except json.JSONDecodeError:
                peers_json = {}
        if isinstance(peers_json, dict):
            self.remote_peers = peers_json

    def _save_peers_to_memory(self) -> None:
        """Persist peer list to SQLite."""
        self.memory.set_policy(GOSSIP_PEERS_KEY, json.dumps(self.remote_peers))

    # ------------------------------------------------------------------
    # HTTP transport: receive gossip message
    # ------------------------------------------------------------------

    async def handle_gossip_message(self, message: dict) -> dict:
        """
        Handle an incoming gossip message from a remote peer.
        Called by the daemon's POST /v1/gossip handler.
        """
        msg_type = message.get("msg_type", "")
        sender = message.get("sender", "")
        fingerprint = message.get("fingerprint", "")

        logger.debug("Received gossip %s from %s", msg_type, sender)

        if msg_type == GossipMessageType.PING.value:
            return await self._handle_ping(sender, fingerprint)
        elif msg_type == GossipMessageType.PUSH_DELTA.value:
            deltas_raw = message.get("deltas", [])
            return await self._handle_push_delta(sender, deltas_raw)
        elif msg_type == GossipMessageType.PULL_REQUEST.value:
            since_seq = message.get("since_seq", 0)
            return await self._handle_pull_request(sender, since_seq)

        return {"msg_type": GossipMessageType.PONG.value, "error": "unknown_type"}

    async def _handle_ping(self, sender: str, remote_fp: str) -> dict:
        """Respond to a PING with our fingerprint and deltas if diverged."""
        my_fp = self.local_peer.state_fingerprint()
        sim = similarity(my_fp, remote_fp) if len(my_fp) == len(remote_fp) else 0.0

        if sim >= 0.999:
            # Match — simple PONG
            return {
                "msg_type": GossipMessageType.PONG.value,
                "sender": self.node_id,
                "receiver": sender,
                "fingerprint": my_fp,
                "diverged": False,
            }

        # Diverged — send our deltas
        my_deltas = [d.to_dict() for d in self.local_peer.stream.deltas]
        return {
            "msg_type": GossipMessageType.PUSH_DELTA.value,
            "sender": self.node_id,
            "receiver": sender,
            "fingerprint": my_fp,
            "deltas": my_deltas,
            "diverged": True,
        }

    async def _handle_push_delta(self, sender: str, deltas_raw: list) -> dict:
        """Apply incoming deltas and acknowledge."""
        applied = 0
        for d_raw in deltas_raw:
            try:
                delta = Delta(
                    agent_id=d_raw.get("agent_id", ""),
                    set_keys=d_raw.get("set_keys", {}),
                    delete_keys=set(d_raw.get("delete_keys", [])),
                    seq=d_raw.get("seq", 0),
                    timestamp_ms=d_raw.get("timestamp_ms", 0),
                    weight=d_raw.get("weight", 1.0),
                    parent_hash=d_raw.get("parent_hash", ""),
                )
                if self.local_peer.receive_delta(delta) is not None:
                    applied += 1
            except Exception as e:
                logger.warning("Failed to apply delta from %s: %s", sender, e)

        self._last_sync[sender] = time.monotonic()
        return {
            "msg_type": GossipMessageType.PONG.value,
            "sender": self.node_id,
            "receiver": sender,
            "fingerprint": self.local_peer.state_fingerprint(),
            "applied": applied,
        }

    async def _handle_pull_request(self, sender: str, since_seq: int) -> dict:
        """Send deltas since a given sequence number."""
        deltas = [d.to_dict() for d in self.local_peer.stream.deltas if d.seq > since_seq]
        return {
            "msg_type": GossipMessageType.PULL_RESPONSE.value,
            "sender": self.node_id,
            "receiver": sender,
            "deltas": deltas,
        }

    # ------------------------------------------------------------------
    # HTTP transport: send gossip message
    # ------------------------------------------------------------------

    async def _send_gossip(self, peer_url: str, payload: dict) -> Optional[dict]:
        """Send a gossip message to a remote peer. Returns response dict or None."""
        url = f"{peer_url}/v1/gossip"
        try:
            timeout = aiohttp.ClientTimeout(total=GOSSIP_TIMEOUT_SECONDS)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        logger.warning("Gossip %s returned %d", url, resp.status)
                        return None
        except aiohttp.ClientError as e:
            logger.debug("Gossip connection error to %s: %s", url, e)
            return None
        except Exception as e:
            logger.debug("Gossip error to %s: %s", url, e)
            return None

    async def initiate_sync_with_peer(self, peer_id: str, peer_url: str) -> dict:
        """Run one gossip round with a specific peer over HTTP."""
        t0 = time.monotonic()

        # Step 1: PING — send our fingerprint
        my_fp = self.local_peer.state_fingerprint()
        ping = {
            "msg_type": GossipMessageType.PING.value,
            "sender": self.node_id,
            "receiver": peer_id,
            "fingerprint": my_fp,
        }

        response = await self._send_gossip(peer_url, ping)
        if response is None:
            return {"peer": peer_id, "status": "unreachable", "latency_ms": 0}

        msg_type = response.get("msg_type", "")
        diverged = response.get("diverged", False)

        if msg_type == GossipMessageType.PONG.value and not diverged:
            # States match — no data exchange needed
            self._last_sync[peer_id] = time.monotonic()
            latency = (time.monotonic() - t0) * 1000
            self.local_peer.stats.fingerprint_hits += 1
            return {"peer": peer_id, "status": "in_sync", "latency_ms": round(latency, 1)}

        # Step 2: Receive deltas from peer
        remote_deltas = response.get("deltas", [])
        if remote_deltas:
            applied = 0
            for d_raw in remote_deltas:
                try:
                    delta = Delta(
                        agent_id=d_raw.get("agent_id", ""),
                        set_keys=d_raw.get("set_keys", {}),
                        delete_keys=set(d_raw.get("delete_keys", [])),
                        seq=d_raw.get("seq", 0),
                        timestamp_ms=d_raw.get("timestamp_ms", 0),
                        weight=d_raw.get("weight", 1.0),
                        parent_hash=d_raw.get("parent_hash", ""),
                    )
                    if self.local_peer.receive_delta(delta) is not None:
                        applied += 1
                except Exception as e:
                    logger.warning("Failed to apply remote delta: %s", e)

            self.local_peer.stats.deltas_propagated += applied

        # Step 3: Push our deltas back
        my_deltas = [d.to_dict() for d in self.local_peer.stream.deltas]
        push = {
            "msg_type": GossipMessageType.PUSH_DELTA.value,
            "sender": self.node_id,
            "receiver": peer_id,
            "fingerprint": self.local_peer.state_fingerprint(),
            "deltas": my_deltas,
        }
        await self._send_gossip(peer_url, push)

        self._last_sync[peer_id] = time.monotonic()
        latency = (time.monotonic() - t0) * 1000
        return {
            "peer": peer_id,
            "status": "synced",
            "latency_ms": round(latency, 1),
            "remote_deltas": len(remote_deltas),
        }

    # ------------------------------------------------------------------
    # Background task
    # ------------------------------------------------------------------

    async def gossip_task(self):
        """Background task: periodically sync with random peers."""
        logger.info("Gossip transport started (node_id=%s)", self.node_id)
        while True:
            await asyncio.sleep(GOSSIP_INTERVAL_SECONDS)
            if not self.remote_peers:
                continue

            # Pick a random peer
            peer_id = random.choice(list(self.remote_peers.keys()))
            peer_url = self.remote_peers[peer_id]

            try:
                result = await self.initiate_sync_with_peer(peer_id, peer_url)
                if result["status"] != "unreachable":
                    logger.debug("Gossip with %s: %s", peer_id, result)
                else:
                    logger.warning("Gossip peer %s unreachable", peer_id)
            except Exception as e:
                logger.warning("Gossip round error: %s", e)

    # ------------------------------------------------------------------
    # State mutations
    # ------------------------------------------------------------------

    def mutate(self, changes: dict[str, str]) -> None:
        """Apply a local mutation — creates a delta that will be gossiped."""
        self.local_peer.mutate(changes)

    def record_request(self, route_decision: dict[str, str]) -> None:
        """Record a routing decision in gossip state for delta propagation."""
        self.mutate({
            "last_route_tier": route_decision.get("tier", ""),
            "last_route_category": route_decision.get("category", ""),
            "last_route_model": route_decision.get("model", ""),
            "last_route_time": str(int(time.time())),
        })

    def status(self) -> dict:
        """Current gossip transport status."""
        return {
            "node_id": self.node_id,
            "fingerprint": self.local_peer.state_fingerprint(),
            "state_keys": len(self.local_peer.state.entries),
            "stream_length": self.local_peer.stream.length,
            "peers": len(self.remote_peers),
            "stats": {
                "rounds": self.local_peer.stats.rounds,
                "fp_hits": self.local_peer.stats.fingerprint_hits,
                "fp_misses": self.local_peer.stats.fingerprint_misses,
                "deltas_propagated": self.local_peer.stats.deltas_propagated,
            },
        }
