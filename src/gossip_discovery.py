"""
Gossip Peer Auto-Discovery via mDNS/DNS-SD.

Discovers other Cortex nodes on the LAN using Avahi/Bonjour,
automatically registers them as gossip peers.

This bridges:
  - avahi-browse (mDNS service discovery)
  - GossipTransport (HTTP-based delta sync)
  - BootTelemetry (boot state as SCL deltas)

When two Cortex USB sticks are on the same network,
they find each other and sync boot optimizations automatically.

Service type: _cortex._tcp
"""

import asyncio
import json
import logging
import subprocess
import time
from typing import Optional

logger = logging.getLogger("cortex.discovery")

SERVICE_TYPE = "_cortex._tcp"
DISCOVERY_INTERVAL = 15  # seconds between scans


class PeerDiscovery:
    """
    mDNS-based peer discovery for Cortex gossip.

    Runs as a background task, scanning for _cortex._tcp services.
    When found, automatically registers them with GossipTransport.
    """

    def __init__(self, gossip_transport, local_port: int = 11411):
        self.gossip = gossip_transport
        self.local_port = local_port
        self._known_peers: dict[str, str] = {}  # id → url
        self._running = False

    async def discovery_task(self):
        """Background task: periodically scan mDNS for Cortex peers."""
        self._running = True
        logger.info("Peer discovery started (service=%s)", SERVICE_TYPE)

        # Register ourselves first
        await self._register_service()

        while self._running:
            await asyncio.sleep(DISCOVERY_INTERVAL)
            try:
                peers = await self._scan_mdns()
                for peer_id, peer_url in peers.items():
                    if peer_id not in self._known_peers:
                        self._known_peers[peer_id] = peer_url
                        self.gossip.add_peer(peer_id, peer_url)
                        logger.info("Discovered peer via mDNS: %s @ %s", peer_id, peer_url)
            except Exception as e:
                logger.debug("mDNS scan failed: %s", e)

    async def _register_service(self):
        """Register our Cortex service via avahi-publish or dns-sd."""
        loop = asyncio.get_event_loop()
        try:
            # Try avahi-publish (Linux)
            await loop.run_in_executor(None, self._avahi_publish)
        except Exception:
            try:
                # Fallback: dns-sd (macOS)
                await loop.run_in_executor(None, self._dnssd_register)
            except Exception as e:
                logger.debug("Service registration failed: %s", e)

    def _avahi_publish(self):
        """Register with avahi-publish-service."""
        import shutil
        avahi_pub = shutil.which("avahi-publish-service")
        if not avahi_pub:
            return
        # Non-blocking — fire and forget
        subprocess.Popen(
            [avahi_pub, "-s", f"cortex-{self.local_port}",
             SERVICE_TYPE, str(self.local_port),
             f"node_id=cortex-{self.local_port}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def _dnssd_register(self):
        """Register with dns-sd (macOS Bonjour)."""
        import shutil
        dns_sd = shutil.which("dns-sd")
        if not dns_sd:
            return
        subprocess.Popen(
            [dns_sd, "-R", f"cortex-{self.local_port}",
             SERVICE_TYPE, "local", str(self.local_port)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    async def _scan_mdns(self) -> dict[str, str]:
        """Scan for _cortex._tcp services. Returns {peer_id: url}."""
        loop = asyncio.get_event_loop()

        # Try avahi-browse first (Linux), then dns-sd (macOS)
        try:
            result = await loop.run_in_executor(None, self._avahi_browse)
            if result:
                return result
        except Exception:
            pass

        try:
            result = await loop.run_in_executor(None, self._dnssd_browse)
            if result:
                return result
        except Exception:
            pass

        return {}

    def _avahi_browse(self) -> dict[str, str]:
        """Use avahi-browse to find peers."""
        import shutil
        avahi = shutil.which("avahi-browse")
        if not avahi:
            return {}

        result = subprocess.run(
            [avahi, "-r", "-t", "-p", SERVICE_TYPE],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return {}

        peers = {}
        for line in result.stdout.splitlines():
            if line.startswith("="):
                parts = line.split(";")
                if len(parts) >= 9:
                    hostname = parts[6]
                    ip = parts[7]
                    port = parts[8]
                    peer_id = f"cortex-{port}"
                    # Don't add ourselves
                    if int(port) != self.local_port or ip not in ("127.0.0.1", "::1"):
                        peers[peer_id] = f"http://{ip}:{port}"
        return peers

    def _dnssd_browse(self) -> dict[str, str]:
        """Use dns-sd -B to find peers (macOS). Limited since dns-sd is interactive."""
        # dns-sd doesn't have a one-shot mode, so we use a timeout approach
        import shutil
        dns_sd = shutil.which("dns-sd")
        if not dns_sd:
            return {}

        # Use a short timeout to grab results
        try:
            result = subprocess.run(
                [dns_sd, "-B", SERVICE_TYPE, "local"],
                capture_output=True, text=True, timeout=3,
            )
        except subprocess.TimeoutExpired:
            # dns-sd runs forever — we kill it and parse what we got
            return {}
        return {}

    def stop(self):
        self._running = False
