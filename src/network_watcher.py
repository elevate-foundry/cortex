"""
Cortex Network Watcher — async background task that monitors interface state.

Detects:
  - Interface coming up (carrier present)
  - Interface going down (carrier lost)
  - IP address acquisition via DHCP
  - Network reattachment after offline boot

Emits SCL records via lifecycle_scl on state changes.
Adjusts inference routing preference (local vs remote) when network changes.

Usage (inside the daemon's asyncio loop):
    from .network_watcher import NetworkWatcher
    watcher = NetworkWatcher()
    asyncio.create_task(watcher.run())
"""

import asyncio
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger("cortex.network")

# Poll interval (seconds) — not too aggressive since this is a background task
POLL_INTERVAL = 5.0


class InterfaceState:
    """Tracks the last-known state of a network interface."""
    __slots__ = ("name", "up", "ip", "carrier")

    def __init__(self, name: str):
        self.name = name
        self.up = False
        self.ip: Optional[str] = None
        self.carrier = False


class NetworkWatcher:
    """
    Async background task that polls network interface state.

    On state change, emits SCL lifecycle records:
      @network → available [interface: eth0, ip: 192.168.1.50]
      @network → lost [interface: eth0, reason: carrier_lost]
      @inference → route [local: primary, remote: optional]
    """

    def __init__(self, poll_interval: float = POLL_INTERVAL):
        self.poll_interval = poll_interval
        self._interfaces: dict[str, InterfaceState] = {}
        self._running = False
        self._has_network = False

    async def run(self) -> None:
        """Main polling loop. Run as asyncio.create_task(watcher.run())."""
        self._running = True
        logger.info("Network watcher started (poll=%.1fs)", self.poll_interval)

        # Initial scan
        await self._scan()

        while self._running:
            await asyncio.sleep(self.poll_interval)
            await self._scan()

    def stop(self) -> None:
        """Signal the watcher to stop."""
        self._running = False

    async def _scan(self) -> None:
        """Scan all non-loopback interfaces for state changes."""
        current_interfaces = self._discover_interfaces()

        for name, new_state in current_interfaces.items():
            old_state = self._interfaces.get(name)

            if old_state is None:
                # First time seeing this interface
                self._interfaces[name] = new_state
                if new_state.carrier and new_state.ip:
                    self._on_network_available(name, new_state.ip)
                continue

            # Check for state transitions
            if new_state.carrier and new_state.ip and not (old_state.carrier and old_state.ip):
                # Interface came up with an IP
                self._on_network_available(name, new_state.ip)
            elif (old_state.carrier or old_state.ip) and not new_state.carrier:
                # Interface went down
                self._on_network_lost(name)

            # Update stored state
            self._interfaces[name] = new_state

        # Check for removed interfaces
        for name in list(self._interfaces.keys()):
            if name not in current_interfaces:
                if self._interfaces[name].carrier:
                    self._on_network_lost(name)
                del self._interfaces[name]

    def _discover_interfaces(self) -> dict[str, InterfaceState]:
        """Discover network interfaces and their state."""
        interfaces: dict[str, InterfaceState] = {}

        # Linux: read /sys/class/net
        net_path = Path("/sys/class/net")
        if net_path.exists():
            for iface_dir in net_path.iterdir():
                name = iface_dir.name
                if name == "lo":
                    continue
                state = InterfaceState(name)
                # Check carrier
                try:
                    carrier = (iface_dir / "carrier").read_text().strip()
                    state.carrier = carrier == "1"
                except Exception:
                    try:
                        operstate = (iface_dir / "operstate").read_text().strip()
                        state.carrier = operstate in ("up", "unknown")
                    except Exception:
                        pass
                # Check operstate for "up"
                try:
                    operstate = (iface_dir / "operstate").read_text().strip()
                    state.up = operstate in ("up", "unknown")
                except Exception:
                    pass
                # Get IP
                state.ip = self._get_interface_ip(name)
                interfaces[name] = state
        else:
            # macOS / fallback: use subprocess
            try:
                result = subprocess.run(
                    ["ifconfig"], capture_output=True, text=True, timeout=3
                )
                if result.returncode == 0:
                    current_iface = None
                    for line in result.stdout.splitlines():
                        if not line.startswith(("\t", " ")):
                            # Interface header line
                            current_iface = line.split(":")[0]
                            if current_iface == "lo0":
                                current_iface = None
                                continue
                            interfaces[current_iface] = InterfaceState(current_iface)
                        elif current_iface and "inet " in line:
                            parts = line.strip().split()
                            idx = parts.index("inet") if "inet" in parts else -1
                            if idx >= 0 and idx + 1 < len(parts):
                                ip = parts[idx + 1]
                                if not ip.startswith("127."):
                                    interfaces[current_iface].ip = ip
                                    interfaces[current_iface].carrier = True
                                    interfaces[current_iface].up = True
                        elif current_iface and "status: active" in line:
                            interfaces[current_iface].carrier = True
                            interfaces[current_iface].up = True
            except Exception:
                pass

        return interfaces

    def _get_interface_ip(self, name: str) -> Optional[str]:
        """Get IPv4 address of a Linux interface."""
        try:
            result = subprocess.run(
                ["ip", "-4", "addr", "show", name],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if "inet " in line:
                        return line.strip().split()[1].split("/")[0]
        except Exception:
            pass
        return None

    def _on_network_available(self, interface: str, ip: str) -> None:
        """Handle network becoming available."""
        logger.info("Network available: %s = %s", interface, ip)
        self._has_network = True
        try:
            from .lifecycle_scl import network_available, inference_route
            network_available(interface, ip)
            # Update inference routing preference
            inference_route(local="primary", remote="optional")
        except Exception as e:
            logger.debug("Failed to emit network SCL: %s", e)

    def _on_network_lost(self, interface: str) -> None:
        """Handle network going down."""
        logger.info("Network lost: %s", interface)
        # Check if we still have any network
        has_any = any(
            s.carrier and s.ip
            for name, s in self._interfaces.items()
            if name != interface
        )
        if not has_any:
            self._has_network = False
        try:
            from .lifecycle_scl import network_lost, inference_route
            network_lost(interface, reason="carrier_lost")
            if not has_any:
                inference_route(local="primary", remote="unavailable")
        except Exception as e:
            logger.debug("Failed to emit network SCL: %s", e)

    @property
    def has_network(self) -> bool:
        """Whether any interface currently has connectivity."""
        return self._has_network

    def status(self) -> dict:
        """Return current network state summary."""
        return {
            "has_network": self._has_network,
            "interfaces": {
                name: {
                    "up": s.up,
                    "carrier": s.carrier,
                    "ip": s.ip,
                }
                for name, s in self._interfaces.items()
            },
        }
