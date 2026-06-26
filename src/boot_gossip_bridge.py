"""
Boot ↔ Gossip Bridge.

Connects the boot telemetry (self-modifying OS) to the gossip transport
(multi-node state sync). When two Cortex sticks are on the same network,
boot optimizations propagate between them via SCL deltas.

Flow:
  1. Stick A boots on Machine X → BootTelemetry emits Delta
  2. Delta stored in local DeltaStream (boot_deltas.scl)
  3. Bridge feeds boot deltas into GossipTransport.local_peer
  4. GossipTransport syncs with Stick B over HTTP
  5. Stick B receives boot deltas → applies them to ITS BootTelemetry
  6. Next boot on Stick B: cached config from Stick A available

The key insight: boot state and inference state use the SAME
Delta/DeltaStream/Peer primitives. This bridge just connects them.
"""

import logging
import time
from typing import Optional, TYPE_CHECKING

from .scl.delta import Delta, SemanticState, DeltaStream
from .scl.types import SCLRecord, SCLDocument, Anchor, Relation, Scope
from .braille.fingerprint import fingerprint as scl_fingerprint, similarity
from .boot_telemetry import BootTelemetry, BOOT_AGENT_ID

if TYPE_CHECKING:
    from .gossip_transport import GossipTransport

logger = logging.getLogger("cortex.boot_gossip")

# Prefix boot state keys to avoid collisions with inference state
BOOT_KEY_PREFIX = "boot."


class BootGossipBridge:
    """
    Bridges boot telemetry deltas into the gossip transport.

    This enables:
      - Boot config propagation between USB sticks
      - Hardware-specific optimizations shared across devices
      - Sneakernet gossip (offline delta exchange via SCL files)
    """

    def __init__(
        self,
        boot_telemetry: BootTelemetry,
        gossip_transport: GossipTransport,
    ):
        self.boot = boot_telemetry
        self.gossip = gossip_transport
        self._last_synced_seq = 0

    def push_boot_state_to_gossip(self) -> int:
        """Push new boot deltas into gossip state.

        Called after boot telemetry logs a new boot or optimizes.
        Returns number of deltas propagated.
        """
        pushed = 0
        for delta in self.boot.stream.deltas:
            if delta.seq <= self._last_synced_seq:
                continue

            # Namespace boot keys to avoid collision with inference state
            prefixed_keys = {
                f"{BOOT_KEY_PREFIX}{k}": v
                for k, v in delta.set_keys.items()
            }

            # Inject into gossip peer's local state
            self.gossip.mutate(prefixed_keys)
            pushed += 1

        self._last_synced_seq = self.boot.stream.length
        if pushed:
            logger.info("Pushed %d boot deltas to gossip state", pushed)
        return pushed

    def pull_boot_state_from_gossip(self) -> int:
        """Extract boot-related deltas from gossip state.

        Called periodically to check if remote peers have sent
        boot optimizations for hardware we haven't seen yet.
        Returns number of deltas applied.
        """
        gossip_state = self.gossip.local_peer.state
        applied = 0

        # Find boot-prefixed keys from gossip state
        boot_keys = {
            k[len(BOOT_KEY_PREFIX):]: v
            for k, v in gossip_state.entries.items()
            if k.startswith(BOOT_KEY_PREFIX)
        }

        if not boot_keys:
            return 0

        # Check if this contains config for hardware we haven't optimized locally
        remote_fp = boot_keys.get("hardware_fp", "")
        local_fp = self.boot.state.entries.get("hardware_fp", "")

        if not remote_fp:
            return 0

        # If we already have this hardware's config, check if remote is better
        if local_fp and remote_fp == local_fp:
            # Same hardware — compare boot times, take the better config
            remote_best = float(boot_keys.get("best_boot_ms", "99999"))
            local_best = float(self.boot.state.entries.get("best_boot_ms", "99999"))

            if remote_best >= local_best:
                return 0  # Our config is already better

            logger.info("Remote peer has better config for this hardware "
                        "(%.0fms vs %.0fms)", remote_best, local_best)

        # Apply remote boot state as a delta
        delta = Delta(
            agent_id=f"{BOOT_AGENT_ID}.gossip",
            set_keys=boot_keys,
            seq=self.boot.stream.length + 1,
            weight=1.5,  # Slightly lower than local optimizations (2.0)
            timestamp_ms=int(time.time() * 1000),
        )
        self.boot.stream.append(delta)
        self.boot._save_delta(delta)
        self.boot._save_state()
        applied += 1

        logger.info("Applied %d boot deltas from gossip (hw_fp=%s)", applied, remote_fp)
        return applied

    def export_for_sneakernet(self) -> str:
        """Export boot state as SCL text for offline transfer.

        Usage: write to a file on the USB stick. When another stick
        mounts this file, it can import the deltas without network.

        Returns SCL document text.
        """
        from .scl.emitter import emit_document
        doc = self.boot.to_scl_document()
        return emit_document(doc)

    def import_from_sneakernet(self, scl_text: str) -> int:
        """Import boot state from an SCL document (offline gossip).

        Usage: read from a file written by another stick's export.
        """
        from .scl.parser import parse_document

        doc = parse_document(scl_text)
        applied = 0

        for record in doc.records:
            if record.relation.verb == "mutate" and record.anchor.name.startswith("cortex.boot"):
                delta = Delta.from_scl(record)
                delta.seq = self.boot.stream.length + 1
                if self.boot.stream.append(delta) is not None:
                    applied += 1

        if applied:
            self.boot._save_state()
            logger.info("Imported %d boot deltas from sneakernet SCL", applied)

        return applied

    def status(self) -> dict:
        """Bridge status."""
        return {
            "boot_deltas": self.boot.stream.length,
            "last_synced_seq": self._last_synced_seq,
            "boot_fingerprint": self.boot.state_fingerprint,
            "gossip_fingerprint": self.gossip.local_peer.state_fingerprint(),
            "gossip_boot_keys": sum(
                1 for k in self.gossip.local_peer.state.entries
                if k.startswith(BOOT_KEY_PREFIX)
            ),
        }
