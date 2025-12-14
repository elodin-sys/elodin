#!/usr/bin/env python3
"""
Component Broadcast Receiver Script

Listens for UDP broadcast component data and writes it to a local Elodin-DB instance.

Usage:
    python3 receive_broadcast.py
    
    # With custom settings
    python3 receive_broadcast.py \
        --db-addr 127.0.0.1:2240 \
        --listen-port 41235 \
        --filter target.world_pos
"""

import argparse
import logging
import signal
import socket
import sys
import time
import threading
from dataclasses import dataclass
from typing import Optional, Dict, List, Set

import numpy as np

# Import protobuf messages
import component_broadcast_pb2 as pb

# Global flag for clean shutdown
_shutdown_requested = False

# Import the Rust-based impeller client
try:
    from impeller_py import ImpellerClient

    IMPELLER_AVAILABLE = True
except ImportError:
    IMPELLER_AVAILABLE = False
    logging.warning("impeller_py not available. Run 'maturin develop' in impeller_py/ first.")


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("receiver")


# Map protobuf PrimType to numpy dtype
PROTO_TO_NUMPY = {
    pb.PRIM_TYPE_U8: np.uint8,
    pb.PRIM_TYPE_U16: np.uint16,
    pb.PRIM_TYPE_U32: np.uint32,
    pb.PRIM_TYPE_U64: np.uint64,
    pb.PRIM_TYPE_I8: np.int8,
    pb.PRIM_TYPE_I16: np.int16,
    pb.PRIM_TYPE_I32: np.int32,
    pb.PRIM_TYPE_I64: np.int64,
    pb.PRIM_TYPE_F32: np.float32,
    pb.PRIM_TYPE_F64: np.float64,
    pb.PRIM_TYPE_BOOL: np.bool_,
}


@dataclass
class ReceivedComponent:
    """Stores information about a received component."""

    source_id: str
    original_name: str
    renamed_name: str
    timestamp_us: int
    values: np.ndarray
    sequence: int
    last_received: float


class ComponentReceiver:
    """Receives component broadcasts over UDP and writes to Elodin-DB."""

    def __init__(
        self,
        db_addr: str = "127.0.0.1:2240",
        listen_port: int = 41235,
        component_filter: Optional[Set[str]] = None,
    ):
        self.db_addr = db_addr
        self.listen_port = listen_port
        self.component_filter = component_filter

        self.running = False

        # UDP receive socket
        self.recv_socket: Optional[socket.socket] = None

        # Impeller client for writing to DB
        self.client: Optional[ImpellerClient] = None
        self._db_available = False

        # Received components: renamed_name -> ReceivedComponent
        self.components: Dict[str, ReceivedComponent] = {}
        self.components_lock = threading.Lock()

        # Known sources
        self.sources: Dict[str, dict] = {}

        # Statistics
        self.packets_received = 0
        self.bytes_received = 0
        self.sequence_gaps = 0
        self.writes_sent = 0
        self.write_errors = 0

    def start(self) -> bool:
        """Start the receiver.

        Starts immediately regardless of DB availability. DB writes will
        be attempted with automatic reconnection.
        """
        # Setup UDP socket
        self.recv_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.recv_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        try:
            self.recv_socket.bind(("0.0.0.0", self.listen_port))
            logger.info(f"Listening on UDP port {self.listen_port}")
        except Exception as e:
            logger.error(f"Failed to bind to port {self.listen_port}: {e}")
            return False

        # Initialize Elodin-DB client if impeller_py is available
        # (does not block on actual connection - writes will auto-reconnect)
        if IMPELLER_AVAILABLE:
            try:
                self.client = ImpellerClient(self.db_addr)
                self.client.connect()
                self._db_available = True
                logger.info(f"Initialized Elodin-DB client for {self.db_addr}")
            except Exception as e:
                logger.warning(f"Could not initialize Elodin-DB client: {e}")
                logger.info("Running in print-only mode (will retry on writes)")
                self.client = None
        else:
            logger.info("Running in print-only mode (impeller_py not available)")

        self.running = True
        return True

    def stop(self):
        """Stop the receiver."""
        self.running = False

        if self.recv_socket:
            self.recv_socket.close()

        if self.client:
            self.client.disconnect()

        logger.info(
            f"Stopped. Received {self.packets_received} packets "
            f"({self.bytes_received} bytes, {self.sequence_gaps} gaps, "
            f"{self.writes_sent} writes, {self.write_errors} write errors)"
        )

    def run(self):
        """Main receive loop."""
        self.recv_socket.settimeout(0.5)

        while self.running:
            try:
                data, addr = self.recv_socket.recvfrom(65535)
                self._handle_packet(data, addr)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    logger.error(f"Receive error: {e}")
                break

    def _handle_packet(self, data: bytes, addr: tuple):
        """Handle a received UDP packet."""
        self.packets_received += 1
        self.bytes_received += len(data)

        # Try to parse as ComponentBroadcast first
        # Must have component_name AND data to be a valid component message
        try:
            msg = pb.ComponentBroadcast()
            msg.ParseFromString(data)
            # Check for actual component data (not just a heartbeat parsed wrong)
            if msg.component_name and len(msg.data) > 0:
                self._handle_component(msg, addr)
                return
        except Exception:
            pass

        # Try to parse as BroadcastHeartbeat
        try:
            msg = pb.BroadcastHeartbeat()
            msg.ParseFromString(data)
            if msg.source_id:
                self._handle_heartbeat(msg, addr)
                return
        except Exception:
            pass

    def _handle_component(self, msg: pb.ComponentBroadcast, addr: tuple):
        """Handle a ComponentBroadcast message."""
        component_name = msg.renamed_component or msg.component_name

        # Apply filter if set
        if self.component_filter and component_name not in self.component_filter:
            return

        # Parse data
        dtype = PROTO_TO_NUMPY.get(msg.data_type, np.float64)
        try:
            values = np.frombuffer(msg.data, dtype=dtype)
            if msg.shape:
                values = values.reshape(msg.shape)
        except Exception as e:
            logger.warning(f"Failed to parse component data: {e}")
            return

        # Check for sequence gaps
        with self.components_lock:
            if component_name in self.components:
                expected_seq = self.components[component_name].sequence + 1
                if msg.sequence != expected_seq and msg.sequence != 0:
                    self.sequence_gaps += 1

            # Store component
            self.components[component_name] = ReceivedComponent(
                source_id=msg.source_id,
                original_name=msg.component_name,
                renamed_name=component_name,
                timestamp_us=msg.timestamp_us,
                values=values,
                sequence=msg.sequence,
                last_received=time.time(),
            )

        # Write to Elodin-DB (handles reconnection if client was unavailable at startup)
        if IMPELLER_AVAILABLE:
            self._write_to_db(component_name, values.astype(np.float64), msg)

    def _handle_heartbeat(self, msg: pb.BroadcastHeartbeat, addr: tuple):
        """Handle a BroadcastHeartbeat message."""
        self.sources[msg.source_id] = {
            "address": addr[0],
            "components": list(msg.components),
            "rate_hz": msg.broadcast_rate_hz,
            "last_seen": time.time(),
        }

    def _write_to_db(self, component_name: str, values: np.ndarray, msg: pb.ComponentBroadcast):
        """Write component data to Elodin-DB.

        Uses persistent connection with automatic reconnection on failure.
        Silently drops writes if DB is unavailable (acceptable for real-time data).
        """
        if not self.client:
            # Try to re-initialize client if it was None
            if IMPELLER_AVAILABLE and not self._db_available:
                try:
                    self.client = ImpellerClient(self.db_addr)
                    self.client.connect()
                    self._db_available = True
                    logger.info(f"Re-established Elodin-DB client for {self.db_addr}")
                except Exception:
                    return  # Silently skip this write
            else:
                return

        try:
            # Send component data using impeller_py (uses persistent connection)
            # Values are already f64, convert to list
            values_list = values.flatten().tolist()

            self.client.send_component(component_name, values_list, msg.timestamp_us)
            self.writes_sent += 1

        except Exception as e:
            self.write_errors += 1
            # Only log occasionally to avoid spam
            if self.write_errors <= 3 or self.write_errors % 100 == 0:
                logger.warning(f"Failed to write to DB (error #{self.write_errors}): {e}")
            # Try to reset the sender connection for next attempt
            try:
                self.client.reset_sender()
            except Exception:
                pass

    def get_component(self, name: str) -> Optional[ReceivedComponent]:
        """Get the latest data for a component."""
        with self.components_lock:
            return self.components.get(name)

    def list_sources(self) -> Dict[str, dict]:
        """List all known broadcast sources."""
        return self.sources.copy()

    def list_components(self) -> List[str]:
        """List all received component names."""
        with self.components_lock:
            return list(self.components.keys())


def main():
    global _shutdown_requested

    parser = argparse.ArgumentParser(
        description="Receive UDP broadcast component data and write to Elodin-DB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python3 receive_broadcast.py

    # With custom settings
    python3 receive_broadcast.py \\
        --db-addr 127.0.0.1:2240 \\
        --listen-port 41235 \\
        --filter target.world_pos

    # Multiple component filter
    python3 receive_broadcast.py --filter target.world_pos --filter target.world_vel
        """,
    )

    parser.add_argument(
        "--db-addr",
        default="127.0.0.1:2240",
        help="Local Elodin-DB address (default: 127.0.0.1:2240)",
    )
    parser.add_argument(
        "--listen-port", type=int, default=41235, help="UDP listen port (default: 41235)"
    )
    parser.add_argument(
        "--filter",
        action="append",
        dest="filters",
        help="Only accept specific component names (can be repeated)",
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

    # Create component filter set
    component_filter = set(args.filters) if args.filters else None

    # Create receiver
    receiver = ComponentReceiver(
        db_addr=args.db_addr,
        listen_port=args.listen_port,
        component_filter=component_filter,
    )

    # Start receiver
    if not receiver.start():
        logger.error("Failed to start receiver")
        sys.exit(1)

    print(f"\nListening for broadcasts on UDP port {args.listen_port}")
    if component_filter:
        print(f"Filtering: {component_filter}")
    print("Press Ctrl+C to stop...\n")

    # Run receiver with periodic status updates
    status_thread = threading.Thread(target=_print_status, args=(receiver,), daemon=True)
    status_thread.start()

    # Run the receiver (will exit when running becomes False)
    try:
        receiver.run()
    finally:
        receiver.stop()


def _print_status(receiver: ComponentReceiver):
    """Print periodic status updates."""
    while receiver.running:
        time.sleep(2)

        sources = receiver.list_sources()
        components = receiver.list_components()

        if sources:
            error_str = f", Errors: {receiver.write_errors}" if receiver.write_errors > 0 else ""
            logger.info(
                f"Sources: {len(sources)}, Components: {len(components)}, "
                f"Packets: {receiver.packets_received}, Gaps: {receiver.sequence_gaps}, "
                f"Writes: {receiver.writes_sent}{error_str}"
            )

            for name in components:
                comp = receiver.get_component(name)
                if comp:
                    age = time.time() - comp.last_received
                    values_preview = comp.values.flatten()[:4]
                    logger.info(
                        f"  {name}: {values_preview}{'...' if comp.values.size > 4 else ''} "
                        f"(age: {age:.2f}s, seq: {comp.sequence})"
                    )


if __name__ == "__main__":
    main()
