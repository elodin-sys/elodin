#!/usr/bin/env python3
"""
Component Broadcast Script

Subscribes to a component from a local Elodin-DB instance and UDP broadcasts
the data at a controlled rate to any listeners on the network.

Usage:
    python3 broadcast_component.py --component bdx.world_pos --rename target.world_pos
    
    # With custom settings
    python3 broadcast_component.py \
        --db-addr 127.0.0.1:2240 \
        --component bdx.world_pos \
        --rename target.world_pos \
        --broadcast-rate 20 \
        --broadcast-port 41235 \
        --source-id bdx-plane
"""

import argparse
import ipaddress
import logging
import signal
import socket
import struct
import sys
import time
import threading
from typing import Optional, List

import netifaces

# Import the Rust-based impeller client
from impeller_py import ImpellerClient, ComponentData

# Import protobuf messages
import component_broadcast_pb2 as pb

# Global flag for clean shutdown
_shutdown_requested = False

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("broadcast")

# Default broadcast addresses (will be augmented dynamically)
BROADCAST_ADDRESSES: List[str] = []


def get_network_broadcast(ip: str) -> Optional[str]:
    """Get the broadcast address for a given IP address using netifaces."""
    try:
        for interface in netifaces.interfaces():
            addrs = netifaces.ifaddresses(interface)
            if netifaces.AF_INET in addrs:
                for addr in addrs[netifaces.AF_INET]:
                    if "addr" in addr and addr["addr"] == ip:
                        if "netmask" in addr:
                            network = ipaddress.IPv4Network(f"{ip}/{addr['netmask']}", strict=False)
                            return str(network.broadcast_address)
        return None
    except Exception as e:
        logger.error(f"Error calculating broadcast address: {e}")
        return None


def get_local_ip() -> Optional[str]:
    """Get the local IP address, with fallback methods."""
    global BROADCAST_ADDRESSES
    fallback_ip = None

    try:
        # First try socket method
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            fallback_ip = ip
            logger.info(f"Found IP using socket method: {ip}")
        except Exception as e:
            logger.debug(f"Socket method failed: {e}")
        finally:
            s.close()

        # Try all interfaces with netifaces
        for interface in netifaces.interfaces():
            addrs = netifaces.ifaddresses(interface)
            if netifaces.AF_INET in addrs:
                for addr in addrs[netifaces.AF_INET]:
                    if "addr" in addr and not addr["addr"].startswith("127."):
                        ip = addr["addr"]
                        broadcast = get_network_broadcast(ip)
                        if broadcast and broadcast not in BROADCAST_ADDRESSES:
                            BROADCAST_ADDRESSES.append(broadcast)
                            logger.info(
                                f"Found interface {interface}: {ip} -> broadcast {broadcast}"
                            )
                        if fallback_ip is None:
                            fallback_ip = ip

        if fallback_ip is not None:
            return fallback_ip

        # Last resort: hostname
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        if not ip.startswith("127."):
            broadcast = get_network_broadcast(ip)
            if broadcast and broadcast not in BROADCAST_ADDRESSES:
                BROADCAST_ADDRESSES.append(broadcast)
            return ip

        logger.warning("Could not find any valid IP address")
        return None
    except Exception as e:
        logger.error(f"Error getting local IP: {e}")
        return None


def get_broadcast_addresses() -> List[str]:
    """Get all broadcast addresses for local network interfaces."""
    global BROADCAST_ADDRESSES

    # Initialize by getting local IP (which populates BROADCAST_ADDRESSES)
    get_local_ip()

    # Always include localhost for same-machine testing
    # (loopback broadcast 127.255.255.255 doesn't always work on macOS)
    if "127.0.0.1" not in BROADCAST_ADDRESSES:
        BROADCAST_ADDRESSES.insert(0, "127.0.0.1")

    if len(BROADCAST_ADDRESSES) <= 1:
        # Only localhost, add fallback addresses
        BROADCAST_ADDRESSES.extend(["192.168.0.255", "192.168.1.255", "10.0.0.255"])
        logger.warning(f"Could not detect network, using fallback: {BROADCAST_ADDRESSES}")

    logger.info(f"Broadcast addresses: {BROADCAST_ADDRESSES}")
    return BROADCAST_ADDRESSES


class ComponentBroadcaster:
    """Broadcasts component data over UDP at a controlled rate."""

    def __init__(
        self,
        db_addr: str = "127.0.0.1:2240",
        component_name: str = "world_pos",
        renamed_component: Optional[str] = None,
        source_id: str = "source",
        broadcast_rate_hz: float = 10.0,
        broadcast_port: int = 41235,
    ):
        self.db_addr = db_addr
        self.component_name = component_name
        self.renamed_component = renamed_component or component_name
        self.source_id = source_id
        self.broadcast_rate_hz = broadcast_rate_hz
        self.broadcast_interval = 1.0 / broadcast_rate_hz
        self.broadcast_port = broadcast_port

        self.running = False
        self.sequence = 0

        # Impeller client (Rust-based)
        self.client: Optional[ImpellerClient] = None

        # UDP broadcast socket
        self.broadcast_socket: Optional[socket.socket] = None
        self.broadcast_addresses: List[str] = []

        # Broadcast thread
        self.broadcast_thread: Optional[threading.Thread] = None

        # Statistics
        self.packets_sent = 0
        self.bytes_sent = 0

    def start(self) -> bool:
        """Start the broadcaster.

        Starts immediately regardless of DB availability. The subscription
        thread will automatically retry connecting to the DB.
        """
        # Setup UDP socket
        self.broadcast_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.broadcast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.broadcast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Get broadcast addresses
        self.broadcast_addresses = get_broadcast_addresses()
        logger.info(f"Broadcasting to: {self.broadcast_addresses}")

        # Create Elodin-DB client (does not block on actual connection)
        try:
            self.client = ImpellerClient(self.db_addr)
            self.client.connect()
            logger.info(f"Initialized Elodin-DB client for {self.db_addr}")
        except Exception as e:
            logger.error(f"Failed to initialize Elodin-DB client: {e}")
            return False

        # Register the component we want to track (works even without discovery)
        try:
            self.client.track_component(self.component_name)
            logger.info(f"Tracking component: {self.component_name}")
        except Exception as e:
            logger.warning(f"Failed to register component: {e}")

        # Subscribe to real-time stream (starts background thread with auto-reconnect)
        # This will automatically retry if DB is not available yet
        try:
            self.client.subscribe_realtime()
            logger.info("Started real-time subscription (will auto-reconnect if needed)")
        except Exception as e:
            logger.error(f"Failed to start subscription: {e}")
            return False

        # Start broadcast thread immediately
        self.running = True
        self.broadcast_thread = threading.Thread(target=self._broadcast_loop, daemon=True)
        self.broadcast_thread.start()

        # Send heartbeat immediately
        self._send_heartbeat()

        logger.info(
            f"Started broadcasting '{self.component_name}' as '{self.renamed_component}' at {self.broadcast_rate_hz} Hz"
        )
        return True

    def stop(self):
        """Stop the broadcaster."""
        self.running = False

        if self.broadcast_thread:
            self.broadcast_thread.join(timeout=2.0)

        if self.client:
            self.client.disconnect()

        if self.broadcast_socket:
            self.broadcast_socket.close()

        logger.info(f"Stopped. Sent {self.packets_sent} packets ({self.bytes_sent} bytes)")

    def _broadcast_loop(self):
        """Main broadcast loop that sends data at the configured rate."""
        last_heartbeat = time.time()
        heartbeat_interval = 1.0
        last_connection_state = None

        while self.running:
            loop_start = time.time()

            # Log connection state changes
            try:
                conn_state = self.client.get_connection_state()
                if conn_state != last_connection_state:
                    logger.info(f"Connection state: {conn_state}")
                    last_connection_state = conn_state
            except Exception:
                pass

            # Get latest data from Rust client
            try:
                data = self.client.get_latest(self.component_name)
                if data:
                    self._broadcast_component(data)
            except Exception as e:
                logger.debug(f"Error getting data: {e}")

            # Send periodic heartbeat
            if time.time() - last_heartbeat >= heartbeat_interval:
                self._send_heartbeat()
                last_heartbeat = time.time()

            # Sleep to maintain broadcast rate
            elapsed = time.time() - loop_start
            sleep_time = self.broadcast_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _broadcast_component(self, data: ComponentData):
        """Broadcast a component data message."""
        # Create protobuf message
        msg = pb.ComponentBroadcast()
        msg.source_id = self.source_id
        msg.component_name = self.component_name
        msg.renamed_component = self.renamed_component
        msg.timestamp_us = data.timestamp
        msg.data_type = pb.PRIM_TYPE_F64  # Rust client converts all to f64
        msg.shape.extend(data.shape)
        # Pack f64 values as bytes
        msg.data = struct.pack(f"<{len(data.values)}d", *data.values)
        msg.sequence = self.sequence

        self.sequence += 1

        # Serialize and broadcast
        serialized = msg.SerializeToString()
        self._send_broadcast(serialized)

        self.packets_sent += 1
        self.bytes_sent += len(serialized)

    def _send_heartbeat(self):
        """Send a heartbeat message."""
        msg = pb.BroadcastHeartbeat()
        msg.source_id = self.source_id
        msg.components.append(self.renamed_component)
        msg.broadcast_rate_hz = self.broadcast_rate_hz
        msg.timestamp_us = int(time.time() * 1_000_000)

        serialized = msg.SerializeToString()
        self._send_broadcast(serialized)

    def _send_broadcast(self, data: bytes):
        """Send data to all broadcast addresses (fire-and-forget)."""
        if not self.broadcast_socket:
            return

        for addr in self.broadcast_addresses:
            try:
                self.broadcast_socket.sendto(data, (addr, self.broadcast_port))
            except Exception:
                # UDP broadcast is fire-and-forget - ignore all errors
                pass

    def get_latest_data(self) -> Optional[ComponentData]:
        """Get the latest component data."""
        if self.client:
            try:
                return self.client.get_latest(self.component_name)
            except Exception:
                return None
        return None


def main():
    global _shutdown_requested

    parser = argparse.ArgumentParser(
        description="Broadcast Elodin-DB component data over UDP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python3 broadcast_component.py --component bdx.world_pos

    # Rename component for the receiver
    python3 broadcast_component.py --component bdx.world_pos --rename target.world_pos

    # Custom settings
    python3 broadcast_component.py \\
        --db-addr 127.0.0.1:2240 \\
        --component bdx.world_pos \\
        --rename target.world_pos \\
        --broadcast-rate 20 \\
        --broadcast-port 41235 \\
        --source-id bdx-plane
        """,
    )

    parser.add_argument(
        "--db-addr", default="127.0.0.1:2240", help="Elodin-DB address (default: 127.0.0.1:2240)"
    )
    parser.add_argument(
        "--component", required=True, help="Component name to subscribe to (e.g., bdx.world_pos)"
    )
    parser.add_argument(
        "--rename", default=None, help="Rename component for broadcast (e.g., target.world_pos)"
    )
    parser.add_argument(
        "--source-id",
        default="source",
        help="Source identifier for this broadcaster (default: source)",
    )
    parser.add_argument(
        "--broadcast-rate", type=float, default=1.0, help="Broadcast rate in Hz (default: 1.0)"
    )
    parser.add_argument(
        "--broadcast-port", type=int, default=41235, help="UDP broadcast port (default: 41235)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Signal handler for clean shutdown - set up FIRST before any blocking calls
    # Note: We use os._exit() because Rust's runtime.block_on() blocks Python's
    # normal signal handling. The Rust code has 5-second timeouts, but for
    # immediate response we force exit.
    def signal_handler(signum, frame):
        global _shutdown_requested
        print("\nShutting down...")
        _shutdown_requested = True
        import os

        os._exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create broadcaster
    broadcaster = ComponentBroadcaster(
        db_addr=args.db_addr,
        component_name=args.component,
        renamed_component=args.rename,
        source_id=args.source_id,
        broadcast_rate_hz=args.broadcast_rate,
        broadcast_port=args.broadcast_port,
    )

    # Start broadcasting
    if not broadcaster.start():
        logger.error("Failed to start broadcaster")
        sys.exit(1)

    print(f"\nBroadcasting '{args.component}' as '{args.rename or args.component}'")
    print(f"Rate: {args.broadcast_rate} Hz, Port: {args.broadcast_port}")
    print("Press Ctrl+C to stop...\n")

    # Main loop with periodic status updates
    while not _shutdown_requested:
        time.sleep(1)
        if _shutdown_requested:
            break
        # Print stats every second
        data = broadcaster.get_latest_data()
        conn_state = broadcaster.client.get_connection_state() if broadcaster.client else "Unknown"
        if data:
            values_preview = data.values[:4]
            suffix = "..." if len(data.values) > 4 else ""
            logger.info(
                f"Latest: {data.name} = {values_preview}{suffix} "
                f"(seq={broadcaster.sequence}, packets={broadcaster.packets_sent})"
            )
        else:
            logger.info(f"Waiting for component data... (connection: {conn_state})")

    broadcaster.stop()
    print("Stopped.")


if __name__ == "__main__":
    main()
