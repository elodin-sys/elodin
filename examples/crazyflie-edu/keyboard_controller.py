"""
Keyboard Controller for Crazyflie Simulation

Provides keyboard input for controlling the drone during simulation.

Key Mappings:
    Q           - Toggle armed state
    Left Shift  - Blue button (dead man switch, hold to enable motors)
    E           - Yellow button
    R           - Green button
    T           - Red button

    WASD        - Left stick (Throttle/Yaw)
        W/S     - Throttle up/down
        A/D     - Yaw left/right

    Arrow Keys  - Right stick (Pitch/Roll)
        Up/Down    - Pitch forward/back
        Left/Right - Roll left/right

Usage:
    from keyboard_controller import KeyboardController

    controller = KeyboardController()
    controller.start()

    # In your loop:
    state = controller.get_state()

    controller.stop()
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

try:
    from pynput import keyboard

    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("Warning: pynput not available. Install with: pip install pynput")


@dataclass
class ControllerState:
    """Current state of the keyboard controller."""

    # Arm state (toggle with Q)
    is_armed: bool = False

    # Buttons (directly mapped to keys)
    button_blue: bool = False  # Left Shift (dead man)
    button_yellow: bool = False  # E
    button_green: bool = False  # R
    button_red: bool = False  # T

    # Left stick (WASD) - values in [-1, 1]
    throttle: float = 0.0  # W/S
    yaw: float = 0.0  # A/D

    # Right stick (Arrows) - values in [-1, 1]
    pitch: float = 0.0  # Up/Down
    roll: float = 0.0  # Left/Right

    # Raw key states for reference
    keys_pressed: set = field(default_factory=set)


class KeyboardController:
    """
    Keyboard controller for the Crazyflie simulation.

    Uses pynput to capture keyboard input in a background thread.
    """

    def __init__(self, on_arm_change: Optional[Callable[[bool], None]] = None):
        """
        Initialize the keyboard controller.

        Args:
            on_arm_change: Optional callback when arm state changes
        """
        self.state = ControllerState()
        self._lock = threading.Lock()
        self._listener: Optional[keyboard.Listener] = None
        self._running = False
        self._on_arm_change = on_arm_change

        if not PYNPUT_AVAILABLE:
            print("KeyboardController: pynput not available, keyboard input disabled")

    def start(self) -> None:
        """Start listening for keyboard input."""
        if not PYNPUT_AVAILABLE:
            print("ERROR: pynput not available - keyboard control disabled")
            return

        if self._running:
            return

        self._running = True
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()

        # Check if listener actually started (may fail due to permissions)
        import time

        time.sleep(0.1)
        if not self._listener.is_alive():
            print("ERROR: Keyboard listener failed to start!")
            print("       On macOS, grant Accessibility permissions to Terminal/Cursor")
            print("       System Preferences > Privacy & Security > Accessibility")
            self._running = False
            return

        print("Keyboard controller started. Press Q to arm/disarm.")
        print("Hold Shift (blue button) + press WASD/Arrows to control.")

    def stop(self) -> None:
        """Stop listening for keyboard input."""
        self._running = False
        if self._listener:
            self._listener.stop()
            self._listener = None

    def get_state(self) -> ControllerState:
        """Get a copy of the current controller state."""
        with self._lock:
            return ControllerState(
                is_armed=self.state.is_armed,
                button_blue=self.state.button_blue,
                button_yellow=self.state.button_yellow,
                button_green=self.state.button_green,
                button_red=self.state.button_red,
                throttle=self.state.throttle,
                yaw=self.state.yaw,
                pitch=self.state.pitch,
                roll=self.state.roll,
                keys_pressed=self.state.keys_pressed.copy(),
            )

    def _on_press(self, key) -> None:
        """Handle key press events."""
        with self._lock:
            key_name = self._get_key_name(key)
            self.state.keys_pressed.add(key_name)

            # Toggle arm with Q
            if key_name == "q":
                self.state.is_armed = not self.state.is_armed
                print(f"Armed: {self.state.is_armed}")
                if self._on_arm_change:
                    self._on_arm_change(self.state.is_armed)

            # Button mappings
            elif key_name == "shift":
                self.state.button_blue = True
            elif key_name == "e":
                self.state.button_yellow = True
            elif key_name == "r":
                self.state.button_green = True
            elif key_name == "t":
                self.state.button_red = True

            # Left stick (WASD)
            elif key_name == "w":
                self.state.throttle = 1.0
            elif key_name == "s":
                self.state.throttle = -1.0
            elif key_name == "a":
                self.state.yaw = -1.0
            elif key_name == "d":
                self.state.yaw = 1.0

            # Right stick (Arrows)
            elif key_name == "up":
                self.state.pitch = 1.0
            elif key_name == "down":
                self.state.pitch = -1.0
            elif key_name == "left":
                self.state.roll = -1.0
            elif key_name == "right":
                self.state.roll = 1.0

    def _on_release(self, key) -> None:
        """Handle key release events."""
        with self._lock:
            key_name = self._get_key_name(key)
            self.state.keys_pressed.discard(key_name)

            # Button releases
            if key_name == "shift":
                self.state.button_blue = False
            elif key_name == "e":
                self.state.button_yellow = False
            elif key_name == "r":
                self.state.button_green = False
            elif key_name == "t":
                self.state.button_red = False

            # Left stick releases (WASD)
            elif key_name == "w" and self.state.throttle > 0:
                self.state.throttle = 0.0
            elif key_name == "s" and self.state.throttle < 0:
                self.state.throttle = 0.0
            elif key_name == "a" and self.state.yaw < 0:
                self.state.yaw = 0.0
            elif key_name == "d" and self.state.yaw > 0:
                self.state.yaw = 0.0

            # Right stick releases (Arrows)
            elif key_name == "up" and self.state.pitch > 0:
                self.state.pitch = 0.0
            elif key_name == "down" and self.state.pitch < 0:
                self.state.pitch = 0.0
            elif key_name == "left" and self.state.roll < 0:
                self.state.roll = 0.0
            elif key_name == "right" and self.state.roll > 0:
                self.state.roll = 0.0

    def _get_key_name(self, key) -> str:
        """Convert pynput key to a simple string name."""
        if not PYNPUT_AVAILABLE:
            return ""

        # Special keys
        if key == keyboard.Key.shift or key == keyboard.Key.shift_l or key == keyboard.Key.shift_r:
            return "shift"
        elif key == keyboard.Key.up:
            return "up"
        elif key == keyboard.Key.down:
            return "down"
        elif key == keyboard.Key.left:
            return "left"
        elif key == keyboard.Key.right:
            return "right"
        elif key == keyboard.Key.space:
            return "space"
        elif key == keyboard.Key.esc:
            return "esc"

        # Regular character keys
        try:
            return key.char.lower() if key.char else ""
        except AttributeError:
            return str(key).replace("Key.", "").lower()


# =============================================================================
# Standalone test
# =============================================================================

if __name__ == "__main__":
    print("Keyboard Controller Test")
    print("=" * 40)
    print("Q: Toggle arm")
    print("Shift: Blue button (hold)")
    print("E/R/T: Yellow/Green/Red buttons")
    print("WASD: Throttle/Yaw")
    print("Arrows: Pitch/Roll")
    print("Ctrl+C to exit")
    print("=" * 40)

    controller = KeyboardController()
    controller.start()

    try:
        while True:
            state = controller.get_state()
            print(
                f"\rArmed:{state.is_armed:d} Blue:{state.button_blue:d} "
                f"Thr:{state.throttle:+.1f} Yaw:{state.yaw:+.1f} "
                f"Pitch:{state.pitch:+.1f} Roll:{state.roll:+.1f}  ",
                end="",
            )
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        controller.stop()
