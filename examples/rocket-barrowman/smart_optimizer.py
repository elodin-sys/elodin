"""
Smart Rocket Optimizer - Full physics and cost coupling.

This optimizer understands that:
- Bigger tubes cost more (more material)
- Bigger tubes = more drag = need more thrust
- Heavier rockets need bigger motors
- There's an optimal size/motor combination for minimum cost

It sweeps through design space to find the CHEAPEST design that meets requirements.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Callable
import re

try:
    import numpy as np
except ImportError:
    np = None

if TYPE_CHECKING:
    from motor_scraper import MotorData

# Import simulation components
from environment import Environment
from motor_model import Motor
from rocket_model import Rocket as RocketModel
from flight_solver import FlightSolver
from openrocket_components import (
    Rocket,
    NoseCone,
    BodyTube,
    TrapezoidFinSet,
    InnerTube,
    Parachute,
    MassComponent,
    MATERIALS,
)
from openrocket_motor import Motor as ORMotor

# Try OpenAI
try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


@dataclass
class Requirements:
    """Parsed requirements."""

    target_altitude_m: float = 3000.0
    altitude_tolerance: float = 0.15  # ±15%
    payload_mass_kg: float = 0.0
    body_diameter_m: Optional[float] = None  # Target body diameter (if specified)
    max_budget_usd: Optional[float] = None
    recovery_type: str = "dual_deploy"  # "single", "dual_deploy", "none"
    min_stability_cal: float = 1.2


@dataclass
class CostBreakdown:
    """Detailed cost breakdown."""

    motor_cost: float = 0.0
    body_tube_cost: float = 0.0
    nose_cone_cost: float = 0.0
    fins_cost: float = 0.0
    motor_mount_cost: float = 0.0
    recovery_cost: float = 0.0
    hardware_cost: float = 0.0  # Fasteners, epoxy, paint
    avionics_cost: float = 0.0

    @property
    def total(self) -> float:
        return sum(
            [
                self.motor_cost,
                self.body_tube_cost,
                self.nose_cone_cost,
                self.fins_cost,
                self.motor_mount_cost,
                self.recovery_cost,
                self.hardware_cost,
                self.avionics_cost,
            ]
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "motor": self.motor_cost,
            "body_tube": self.body_tube_cost,
            "nose_cone": self.nose_cone_cost,
            "fins": self.fins_cost,
            "motor_mount": self.motor_mount_cost,
            "recovery": self.recovery_cost,
            "hardware": self.hardware_cost,
            "avionics": self.avionics_cost,
            "total": self.total,
        }


@dataclass
class DesignPoint:
    """A single design point in the optimization space."""

    body_diameter_m: float
    body_length_m: float
    motor_designation: str
    motor_impulse_ns: float

    # Results (filled after simulation)
    simulated_altitude_m: float = 0.0
    dry_mass_kg: float = 0.0
    cost: Optional[CostBreakdown] = None
    meets_target: bool = False
    confidence: float = 0.0

    rocket_config: Optional[Dict] = None
    motor_config: Optional[Dict] = None


class SmartOptimizer:
    """
    Smart optimizer that sweeps design space to find cheapest working design.

    Key insight: There's a coupled relationship between:
    - Body size → material cost, weight, drag
    - Motor size → thrust, cost, weight
    - Target altitude → determines minimum motor/body combo

    We sweep through body sizes and motors to find the sweet spot.
    """

    # Standard body tube sizes (ID in meters, with typical costs per meter in USD)
    TUBE_SIZES = [
        {"name": "29mm", "id": 0.029, "od": 0.032, "cost_per_m": 8},
        {"name": "38mm", "id": 0.038, "od": 0.041, "cost_per_m": 12},
        {"name": "54mm", "id": 0.054, "od": 0.058, "cost_per_m": 18},
        {"name": "75mm", "id": 0.075, "od": 0.080, "cost_per_m": 28},
        {"name": "98mm", "id": 0.098, "od": 0.104, "cost_per_m": 40},
        {"name": "4in", "id": 0.102, "od": 0.108, "cost_per_m": 45},
        {"name": "5in", "id": 0.127, "od": 0.133, "cost_per_m": 60},
        {"name": "6in", "id": 0.152, "od": 0.159, "cost_per_m": 80},
    ]

    # Motor costs by impulse class
    MOTOR_COSTS = {
        "A": 6,
        "B": 8,
        "C": 10,
        "D": 12,
        "E": 18,
        "F": 25,
        "G": 40,
        "H": 65,
        "I": 95,
        "J": 140,
        "K": 200,
        "L": 280,
        "M": 400,
        "N": 600,
        "O": 900,
    }

    # Material costs (USD per kg)
    MATERIAL_COST_PER_KG = {
        "Fiberglass": 45,
        "Carbon fiber": 120,
        "Cardboard": 8,
        "Blue tube": 25,
        "Phenolic": 35,
    }

    def __init__(self, motor_database: List["MotorData"] = None, openai_api_key: str = None):
        self.motors = motor_database or []
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.use_openai = OPENAI_AVAILABLE and self.openai_api_key

        if self.use_openai:
            self.client = OpenAI(api_key=self.openai_api_key)

        # Pre-sort motors by impulse
        self.motors_by_impulse = sorted(
            [m for m in self.motors if m.total_impulse > 0], key=lambda m: m.total_impulse
        )

    def parse_requirements(self, text: str) -> Requirements:
        """Parse natural language to requirements."""
        req = Requirements()
        text_lower = text.lower()

        # Parse altitude
        patterns = [
            (r"(\d+)\s*(?:thousand|k)\s*(?:ft|feet)", lambda m: float(m.group(1)) * 1000 * 0.3048),
            (r"(\d+)\s*(?:ft|feet)", lambda m: float(m.group(1)) * 0.3048),
            (r"(\d+)\s*(?:m|meters?)\b", lambda m: float(m.group(1))),
            (r"(\d+)\s*km", lambda m: float(m.group(1)) * 1000),
        ]
        for pattern, converter in patterns:
            match = re.search(pattern, text_lower)
            if match:
                req.target_altitude_m = converter(match)
                break

        # Parse payload/weight
        # "weighs 50lbs" means total weight, not just payload
        # "50lb payload" means payload only
        weight_patterns = [
            (
                r"weighs\s+(\d+\.?\d*)\s*(?:lb|lbs)",
                lambda m: float(m.group(1)) * 0.4536,
            ),  # Total weight
            (r"weighs\s+(\d+\.?\d*)\s*kg", lambda m: float(m.group(1))),  # Total weight
            (
                r"(\d+\.?\d*)\s*(?:lb|lbs)\s+payload",
                lambda m: float(m.group(1)) * 0.4536,
            ),  # Payload only
            (r"(\d+\.?\d*)\s*kg\s+payload", lambda m: float(m.group(1))),  # Payload only
            (
                r"(\d+\.?\d*)\s*(?:lb|lbs)",
                lambda m: float(m.group(1)) * 0.4536,
            ),  # Default: assume payload
            (r"(\d+\.?\d*)\s*kg", lambda m: float(m.group(1))),  # Default: assume payload
        ]
        for pattern, converter in weight_patterns:
            match = re.search(pattern, text_lower)
            if match:
                mass_kg = converter(match)
                # If "weighs" is in the pattern, it's total weight, so estimate payload
                if "weighs" in pattern:
                    # Total weight includes structure, so payload is ~60-70% of total
                    req.payload_mass_kg = mass_kg * 0.65
                else:
                    req.payload_mass_kg = mass_kg
                break

        # Parse body diameter/airframe size
        diameter_patterns = [
            (
                r"(\d+\.?\d*)\s*(?:in|inch|inches)",
                lambda m: float(m.group(1)) * 0.0254,
            ),  # inches to meters
            (
                r"(\d+\.?\d*)\s*(?:mm|millimeters?)",
                lambda m: float(m.group(1)) * 0.001,
            ),  # mm to meters
            (
                r"(\d+\.?\d*)\s*(?:cm|centimeters?)",
                lambda m: float(m.group(1)) * 0.01,
            ),  # cm to meters
        ]
        for pattern, converter in diameter_patterns:
            match = re.search(pattern, text_lower)
            if match:
                req.body_diameter_m = converter(match)
                break

        # Parse budget
        match = re.search(r"\$\s*(\d+)", text)
        if match:
            req.max_budget_usd = float(match.group(1))

        # Recovery type
        if "dual" in text_lower or "drogue" in text_lower:
            req.recovery_type = "dual_deploy"
        elif "single" in text_lower or "one chute" in text_lower:
            req.recovery_type = "single"

        return req

    def _get_tube_for_motor(self, motor_diameter: float, prefer_smaller: bool = True) -> Dict:
        """Get tube that fits a motor.

        Args:
            motor_diameter: Motor diameter in meters
            prefer_smaller: If True, return smallest fitting tube. If False, return largest fitting tube.
        """
        fitting_tubes = []
        for tube in self.TUBE_SIZES:
            if tube["id"] >= motor_diameter + 0.003:  # 3mm clearance
                fitting_tubes.append(tube)

        if not fitting_tubes:
            return self.TUBE_SIZES[-1]  # Largest if none fit

        if prefer_smaller:
            return fitting_tubes[0]  # Smallest fitting tube
        else:
            return fitting_tubes[-1]  # Largest fitting tube

    def _estimate_motor_cost(self, motor) -> float:
        """Estimate motor cost from class."""
        if not motor.designation:
            return 100
        motor_class = motor.designation[0].upper()
        return self.MOTOR_COSTS.get(motor_class, 100)

    def _calculate_costs(
        self,
        tube: Dict,
        body_length: float,
        nose_length: float,
        motor,
        recovery_type: str,
    ) -> CostBreakdown:
        """Calculate full build cost."""
        cost = CostBreakdown()

        # Motor
        cost.motor_cost = self._estimate_motor_cost(motor)

        # Body tube
        cost.body_tube_cost = tube["cost_per_m"] * body_length

        # Nose cone (roughly 2x tube cost per unit length)
        cost.nose_cone_cost = tube["cost_per_m"] * nose_length * 1.5

        # Fins (4 fins, cost scales with tube diameter)
        fin_cost_factor = (tube["od"] / 0.05) ** 1.5  # Bigger = more expensive
        cost.fins_cost = 15 * fin_cost_factor

        # Motor mount
        cost.motor_mount_cost = 10 + (motor.diameter * 100)  # Scales with motor size

        # Recovery
        if recovery_type == "dual_deploy":
            cost.recovery_cost = 80  # Main + drogue + altimeter
            cost.avionics_cost = 100  # Altimeter + battery + charges
        elif recovery_type == "single":
            cost.recovery_cost = 40  # Just main chute
            cost.avionics_cost = 0
        else:
            cost.recovery_cost = 0
            cost.avionics_cost = 0

        # Hardware (epoxy, fasteners, paint, rail buttons, etc.)
        # Scales with rocket size
        size_factor = tube["od"] * body_length * 100
        cost.hardware_cost = 20 + size_factor * 2

        return cost

    def _estimate_dry_mass(
        self,
        tube: Dict,
        body_length: float,
        nose_length: float,
        motor,
        payload_mass: float,
        recovery_type: str,
    ) -> float:
        """Estimate dry mass of rocket."""
        # Fiberglass tube: ~1850 kg/m³, wall ~2-3mm
        tube_volume = math.pi * ((tube["od"] / 2) ** 2 - (tube["id"] / 2) ** 2) * body_length
        tube_mass = tube_volume * 1850

        # Nose (solid fiberglass shell)
        nose_volume = 0.5 * math.pi * (tube["od"] / 2) ** 2 * nose_length * 0.1  # ~10% solid
        nose_mass = nose_volume * 1850

        # Fins (4 fins)
        fin_span = tube["od"] * 0.8
        fin_chord = tube["od"] * 1.0
        fin_thickness = 0.004
        fin_mass = 4 * 0.5 * fin_chord * fin_span * fin_thickness * 1850

        # Motor mount + centering rings
        mount_mass = 0.05 + motor.diameter * 0.5

        # Recovery system
        if recovery_type == "dual_deploy":
            recovery_mass = 0.3  # Main + drogue + altimeter + hardware
        elif recovery_type == "single":
            recovery_mass = 0.15
        else:
            recovery_mass = 0.0

        # Hardware
        hardware_mass = 0.1 + (tube["od"] * body_length * 5)

        total = (
            tube_mass
            + nose_mass
            + fin_mass
            + mount_mass
            + recovery_mass
            + hardware_mass
            + payload_mass
        )
        return total

    def _build_and_simulate(
        self,
        tube: Dict,
        motor,
        requirements: Requirements,
    ) -> Tuple[float, float, Dict, Dict]:
        """Build rocket and run simulation. Returns (altitude, dry_mass, rocket_config, motor_config)."""

        # Calculate dimensions - use Calisto-like proportions for stability
        body_radius = tube["od"] / 2
        nose_length = body_radius * 4  # 4:1 fineness (like Calisto)

        # Body length calculation - ensure enough space for motor, payload, and structure
        motor_bay_length = motor.length + 0.1

        # Payload bay calculation - but cap it to prevent absurdly long rockets
        # For heavy payloads, we need larger tubes, not longer tubes
        payload_volume_needed = requirements.payload_mass_kg / 500.0  # m³ (typical payload density)
        payload_bay_length = payload_volume_needed / (math.pi * body_radius**2)
        # Cap payload bay to reasonable length (max 2m for any payload)
        # If payload doesn't fit, we need a larger tube (handled by tube selection)
        payload_bay_length = min(max(payload_bay_length, 0.2), 2.0)  # 20cm to 2m

        # Minimum body length: ensure stability (L/D ratio)
        # Calisto has body_length = 1.90m for 127mm diameter = 15× diameter
        # Use similar ratio: 12-18× diameter for stability
        min_body_length_by_diameter = body_radius * 2 * 12  # 12× diameter minimum
        min_body_length_by_motor = motor.length + 0.5  # Motor + clearance

        # Functional length: motor bay + payload bay + avionics/recovery space
        functional_length = motor_bay_length + payload_bay_length + 0.3

        body_length = max(
            functional_length,  # Functional length
            min_body_length_by_diameter,  # Stability requirement
            min_body_length_by_motor,  # Motor requirement
        )

        # Cap body length to prevent absurdly long rockets (max L/D of 40)
        total_length = nose_length + body_length
        max_total_length = body_radius * 2 * 40  # Max L/D of 40
        if total_length > max_total_length:
            # Cap at max L/D, but ensure minimum functional length
            body_length = min(
                max_total_length - nose_length,
                max(functional_length, min_body_length_by_motor),  # Keep functional minimum
            )

        # Validate dimensions before building
        if body_radius <= 0 or nose_length <= 0 or body_length <= 0:
            print(
                f"Warning: Invalid dimensions for {motor.designation}: radius={body_radius}, nose={nose_length}, body={body_length}"
            )
            return (0.0, 0.0, {}, {})

        if motor.diameter > tube["id"]:
            print(
                f"Warning: Motor {motor.designation} (diameter={motor.diameter * 1000:.1f}mm) too large for tube (ID={tube['id'] * 1000:.1f}mm)"
            )
            return (0.0, 0.0, {}, {})

        # Build rocket
        rocket = Rocket("Optimized")

        nose = NoseCone(
            name="Nose",
            length=nose_length,
            base_radius=body_radius,
            thickness=0.003,
            shape=NoseCone.Shape.VON_KARMAN,
        )
        nose.material = MATERIALS["Fiberglass"]
        rocket.add_child(nose)

        body = BodyTube(name="Body", length=body_length, outer_radius=body_radius, thickness=0.003)
        body.material = MATERIALS["Fiberglass"]
        body.position.x = nose_length
        rocket.add_child(body)

        # Fins - use Calisto-like proportions for proven stability
        # Calisto: root=0.120m (1.89× radius), tip=0.060m (50% of root), span=0.110m (1.73× radius)
        # Use similar ratios but scale with body radius
        fin_span = max(body_radius * 1.7, 0.03)  # ~1.7× radius (like Calisto)
        fin_root = max(body_radius * 1.9, 0.04)  # ~1.9× radius (like Calisto)
        fin_root = min(fin_root, body_length * 0.20)  # Not more than 20% of body length
        fin_tip = max(fin_root * 0.5, 0.015)  # 50% of root (like Calisto)
        # Ensure fin doesn't extend beyond body
        if fin_root > body_length * 0.25:
            fin_root = body_length * 0.25
            fin_tip = fin_root * 0.5

        fins = TrapezoidFinSet(
            name="Fins",
            fin_count=4,
            root_chord=fin_root,
            tip_chord=fin_tip,
            span=fin_span,
            sweep=fin_root * 0.3,
            thickness=0.004,
        )
        fins.material = MATERIALS["Fiberglass"]
        # Fins position is relative to body start, not absolute
        # Position at aft end with some clearance
        fin_position = max(0.0, body_length - fin_root - 0.01)
        fins.position.x = fin_position
        body.add_child(fins)

        # Motor mount - ensure it fits and is positioned correctly
        mount_length = motor.length + 0.05
        mount_outer_radius = motor.diameter / 2 + 0.005
        # Ensure mount fits inside body (with clearance)
        if mount_outer_radius >= body_radius - 0.01:
            mount_outer_radius = body_radius - 0.01
            if mount_outer_radius <= motor.diameter / 2:
                # Motor won't fit - this should have been caught earlier, but double-check
                print(f"Warning: Motor {motor.designation} won't fit in body tube")
                return (0.0, 0.0, {}, {})

        mount = InnerTube(
            name="Mount",
            length=mount_length,
            outer_radius=mount_outer_radius,
            thickness=0.003,
        )
        mount.motor_mount = True
        # Motor mount position is relative to body start
        # Position at aft end, ensuring it doesn't extend beyond body
        mount_position = max(0.0, body_length - mount_length - 0.01)
        mount.position.x = mount_position
        body.add_child(mount)

        # Estimate dry mass properly - structure + payload
        # First estimate structure mass (without payload)
        structure_mass = self._estimate_dry_mass(
            tube,
            body_length,
            nose_length,
            motor,
            0.0,  # Don't include payload in structure estimate
            requirements.recovery_type,
        )

        # Add payload as actual MassComponent if specified
        if requirements.payload_mass_kg > 0:
            # Payload bay position - in forward section of body
            payload_position = nose_length + body_length * 0.3  # 30% into body
            payload = MassComponent(
                name="Payload",
                mass=requirements.payload_mass_kg,
                length=0.15,  # Typical payload bay length
                radius=body_radius * 0.9,  # Slightly smaller than body
            )
            payload.position.x = payload_position - nose_length  # Relative to body start
            body.add_child(payload)

        # Total dry mass = structure + payload
        dry_mass = structure_mass + requirements.payload_mass_kg
        landed_mass = dry_mass + motor.case_mass

        # Size for 5 m/s descent - ensure reasonable size
        main_area = (2 * landed_mass * 9.81) / (1.225 * 25 * 1.5)
        main_dia = math.sqrt(4 * main_area / math.pi)
        # Ensure parachute is reasonable size (not too small, not too large)
        main_dia = max(main_dia, 0.3)  # At least 30cm
        main_dia = min(main_dia, 5.0)  # Not more than 5m (unrealistic for model rockets)

        main = Parachute(name="Main", diameter=main_dia, cd=1.5)
        if requirements.recovery_type == "dual_deploy":
            main.deployment_event = "ALTITUDE"
            main.deployment_altitude = 300
        else:
            main.deployment_event = "APOGEE"
        main.deployment_delay = 1.0
        nose.add_child(main)

        if requirements.recovery_type == "dual_deploy":
            drogue_area = (2 * landed_mass * 9.81) / (0.8 * 900 * 1.3)
            drogue_dia = math.sqrt(4 * drogue_area / math.pi)
            # Ensure drogue is reasonable size
            drogue_dia = max(drogue_dia, 0.2)  # At least 20cm
            drogue_dia = min(drogue_dia, 2.0)  # Not more than 2m
            drogue = Parachute(name="Drogue", diameter=drogue_dia, cd=1.3)
            drogue.deployment_event = "APOGEE"
            drogue.deployment_delay = 1.0
            nose.add_child(drogue)

        # Validate rocket before calculating reference values
        try:
            rocket.calculate_reference_values()
        except Exception as e:
            print(f"Warning: Failed to calculate reference values for {motor.designation}: {e}")
            return (0.0, 0.0, {}, {})

        # Verify rocket has valid reference values
        if rocket.reference_diameter <= 0 or rocket.reference_length <= 0:
            print(f"Warning: Invalid reference values for {motor.designation}")
            return (0.0, 0.0, {}, {})

        # Quick stability check: ensure rocket has reasonable length-to-diameter ratio
        # Too short rockets are unstable, too long are inefficient but can still work
        l_over_d = rocket.reference_length / rocket.reference_diameter
        if l_over_d < 5.0:
            # Rocket too short - likely unstable
            print(
                f"Warning: Rocket too short (L/D={l_over_d:.1f}) for {motor.designation}, likely unstable"
            )
            return (0.0, 0.0, {}, {})
        if l_over_d > 40.0:
            # Rocket very long - might have structural issues, but allow up to 40
            # Don't reject, just note it - some rockets can work with high L/D
            pass

        # Build motor object
        motor_obj = ORMotor(
            designation=motor.designation,
            manufacturer=motor.manufacturer,
            diameter=motor.diameter,
            length=motor.length,
            total_mass=motor.total_mass,
            propellant_mass=motor.propellant_mass,
            thrust_curve=motor.thrust_curve,
            burn_time=motor.burn_time,
            total_impulse=motor.total_impulse,
        )

        # Run simulation - must run full flight to get accurate results
        try:
            # Validate rocket before building
            if body_length <= 0 or body_radius <= 0 or nose_length <= 0:
                print(f"Warning: Invalid rocket dimensions for {motor.designation}")
                altitude = 0.0
            else:
                rocket_model = RocketModel(rocket)
                motor_wrapped = Motor.from_openrocket(motor_obj)
                env = Environment()

                # Validate rocket model before creating solver
                try:
                    # Quick sanity check on rocket geometry
                    if rocket_model.reference_diameter <= 0 or rocket_model.reference_length <= 0:
                        print(f"Warning: Invalid rocket geometry for {motor.designation}")
                        altitude = 0.0
                    else:
                        solver = FlightSolver(
                            rocket=rocket_model,
                            motor=motor_wrapped,
                            environment=env,
                            rail_length=3.0,
                            inclination_deg=5.0,
                            dt=0.02,
                        )
                        # Run simulation to impact (or max 600s) - don't use short max_time
                        # The simulation will stop at impact automatically or if it detects failure
                        result = solver.run()
                except Exception as e:
                    print(f"Warning: Failed to create solver for {motor.designation}: {e}")
                    altitude = 0.0
                    result = None

                if result is not None and result.history and len(result.history) > 0:
                    # Get all valid altitudes (filter out NaN/Inf)
                    if np is not None:
                        altitudes = [
                            s.position[2] for s in result.history if np.isfinite(s.position[2])
                        ]
                    else:
                        # Fallback if numpy not available
                        altitudes = [
                            s.position[2] for s in result.history if abs(s.position[2]) < 1e10
                        ]

                    if altitudes:
                        altitude = max(altitudes)

                        # FIRST: Check if altitude is at clamp boundary (indicates simulation failure)
                        # The flight solver clamps positions to ±1e5 (100000m), so if we see close to 1e5, it's a clamp
                        # Check within 1km of clamp boundary to catch values like 99938m, 99974m, etc.
                        if abs(altitude - 1e5) < 1000.0 or abs(altitude + 1e5) < 1000.0:
                            # Altitude is very close to clamp boundary - simulation failed
                            # Silently mark as failure (no warning, too verbose for optimizer)
                            altitude = 0.0
                        # Sanity check: if altitude is unreasonably high, something is wrong
                        # 50km is already very high for model rockets, 100km is definitely wrong
                        elif altitude > 50000:  # 50km is already very high for model rockets
                            # Likely simulation error - check if simulation actually ran
                            # If all altitudes are the same or increasing linearly, it's probably wrong
                            if len(altitudes) > 10:
                                # Check if trajectory looks reasonable
                                alt_diff = max(altitudes) - min(altitudes)
                                if alt_diff < 10:  # No altitude change
                                    print(
                                        f"Warning: No altitude change for {motor.designation}, simulation may have failed"
                                    )
                                    altitude = 0.0
                                else:
                                    # Check if many altitudes are at the clamp (indicates widespread failure)
                                    clamped_count = sum(
                                        1
                                        for alt in altitudes
                                        if abs(alt - 1e5) < 1.0 or abs(alt + 1e5) < 1.0
                                    )
                                    if (
                                        clamped_count > len(altitudes) * 0.1
                                    ):  # More than 10% clamped
                                        print(
                                            f"Warning: Many positions clamped for {motor.designation} - simulation failed"
                                        )
                                        altitude = 0.0
                                    else:
                                        print(
                                            f"Warning: Invalid altitude {altitude:.0f}m for {motor.designation}, treating as failure"
                                        )
                                        altitude = 0.0
                            else:
                                print(
                                    f"Warning: Invalid altitude {altitude:.0f}m for {motor.designation}, treating as failure"
                                )
                                altitude = 0.0
                        elif np is not None and not np.isfinite(altitude):
                            print(
                                f"Warning: Non-finite altitude for {motor.designation}, treating as failure"
                            )
                            altitude = 0.0
                        else:
                            # Valid altitude - check if simulation ran long enough
                            if len(result.history) < 10:
                                print(
                                    f"Warning: Very short simulation ({len(result.history)} points) for {motor.designation}"
                                )
                                altitude = 0.0
                            # Check if final altitude is reasonable (should be near ground after impact)
                            final_alt = result.history[-1].position[2]
                            if final_alt > 1000 and len(result.history) > 100:
                                # Simulation didn't reach impact - might be stuck
                                print(
                                    f"Warning: Simulation didn't reach impact (final alt={final_alt:.0f}m) for {motor.designation}"
                                )
                                # Still use the max altitude if it's reasonable
                                if altitude > 50000:
                                    altitude = 0.0
                    else:
                        # All altitudes were NaN/Inf
                        print(
                            f"Warning: All altitudes invalid for {motor.designation}, treating as failure"
                        )
                        altitude = 0.0
                else:
                    print(f"Warning: No simulation history for {motor.designation}")
                    altitude = 0.0
        except Exception as e:
            # Log the error for debugging but don't crash
            import traceback

            error_msg = str(e)
            print(f"Simulation error for {motor.designation}: {error_msg}")
            print(
                f"  Rocket: {body_length:.2f}m body, {body_radius * 2 * 1000:.1f}mm diameter, {nose_length:.2f}m nose"
            )
            print(
                f"  Motor: {motor.designation}, {motor.total_impulse:.0f} N·s, {motor.diameter * 1000:.1f}mm"
            )
            # Only print full traceback for unexpected errors, not validation errors
            if "Invalid" not in error_msg and "dimension" not in error_msg.lower():
                traceback.print_exc()
            altitude = 0.0

        # Build configs
        rocket_config = {
            "name": "Optimized",
            "nose_length": nose_length,
            "nose_shape": "VON_KARMAN",
            "body_length": body_length,
            "body_radius": body_radius,
            "fin_count": 4,
            "fin_root_chord": fin_root,
            "fin_tip_chord": fin_tip,
            "fin_span": fin_span,
            "fin_sweep": fin_root * 0.3,
            "has_main_chute": True,
            "main_chute_diameter": main_dia,
            "has_drogue": requirements.recovery_type == "dual_deploy",
            "drogue_diameter": drogue_dia if requirements.recovery_type == "dual_deploy" else 0,
        }

        motor_config = {
            "designation": motor.designation,
            "manufacturer": motor.manufacturer,
            "total_impulse": motor.total_impulse,
            "diameter": motor.diameter,
            "length": motor.length,
            "total_mass": motor.total_mass,
            "thrust_curve": motor.thrust_curve,
        }

        return altitude, dry_mass, rocket_config, motor_config

    def optimize(
        self,
        requirements: Requirements,
        max_iterations: int = 30,
        callback: Callable = None,
    ) -> Tuple[Optional[DesignPoint], List[DesignPoint], List[str]]:
        """
        Find cheapest design that meets requirements.

        Strategy:
        1. Start with smallest tube that fits the smallest viable motor
        2. Sweep through motors from small to large
        3. For each motor, check if altitude is within tolerance
        4. Track cheapest working design
        5. Stop early if we find a cheap solution

        Returns: (best_design, all_designs, log)
        """
        log = []
        designs = []
        best = None
        best_cost = float("inf")

        target = requirements.target_altitude_m
        tol = requirements.altitude_tolerance
        min_alt = target * (1 - tol)
        max_alt = target * (1 + tol)

        log.append(f"Target: {target:.0f}m ± {tol * 100:.0f}% ({min_alt:.0f}m - {max_alt:.0f}m)")
        log.append(f"Payload: {requirements.payload_mass_kg:.1f}kg")
        if requirements.max_budget_usd:
            log.append(f"Budget: ${requirements.max_budget_usd:.0f}")

        # Estimate impulse range we need
        # Altitude/impulse ratio depends heavily on rocket mass and drag:
        # - Light rockets (<5kg): ~1.0-2.0 m/Ns
        # - Medium rockets (5-15kg): ~0.5-1.0 m/Ns
        # - Heavy rockets (15-30kg): ~0.2-0.5 m/Ns
        # - Very heavy rockets (>30kg): ~0.1-0.3 m/Ns
        #
        # Account for payload mass and expected dry mass
        # For a 50lb rocket, if "weighs 50lbs" means total weight:
        # - Total weight: 22.7kg
        # - Payload: ~15kg (65% of total)
        # - Structure: ~7kg (35% of total)
        # - Motor: ~3-5kg (varies by motor)
        # - Total at launch: ~25-30kg
        #
        # If "50lb payload" means payload only:
        # - Payload: 22.7kg
        # - Structure: ~10-15kg (scales with size)
        # - Motor: ~3-8kg
        # - Total at launch: ~35-45kg

        # Estimate structure mass based on payload
        if requirements.payload_mass_kg > 15:
            # Heavy rocket - structure is ~40-50% of payload for large rockets
            estimated_structure_mass = requirements.payload_mass_kg * 0.45
        elif requirements.payload_mass_kg > 5:
            # Medium rocket - structure is ~50-60% of payload
            estimated_structure_mass = requirements.payload_mass_kg * 0.55
        else:
            # Light rocket - structure is ~60-80% of payload
            estimated_structure_mass = max(2.0, requirements.payload_mass_kg * 0.7)

        # Estimate motor mass (varies by impulse class)
        # For K/L/M motors: 3-8kg typical
        estimated_motor_mass = 5.0  # Average for K/L/M motors

        estimated_total_mass_kg = (
            requirements.payload_mass_kg + estimated_structure_mass + estimated_motor_mass
        )

        # Altitude/impulse ratio - calibrated from Calisto (19kg loaded, 6026 N·s, 3350m = 0.556 m/N·s)
        # Real-world data shows heavier rockets get MORE altitude per impulse (better mass ratio)
        if estimated_total_mass_kg < 5:
            altitude_per_impulse = 1.2  # Light rocket (was 1.5, too optimistic)
        elif estimated_total_mass_kg < 15:
            altitude_per_impulse = 0.7  # Medium rocket (was 0.8, too optimistic)
        elif estimated_total_mass_kg < 30:
            altitude_per_impulse = (
                0.55  # Heavy rocket - Calisto is 0.556 at 19kg (was 0.4, too low!)
            )
        else:
            altitude_per_impulse = 0.45  # Very heavy rocket (was 0.25, way too low!)

        # Add drag penalty for larger diameters (6" = 152mm is large)
        # Larger diameter = more drag = lower altitude per impulse
        if requirements.body_diameter_m and requirements.body_diameter_m > 0.1:  # >100mm
            # 6" = 152mm, moderate drag penalty
            if requirements.body_diameter_m > 0.15:  # >6"
                altitude_per_impulse *= 0.85  # 15% penalty for very large
            else:
                altitude_per_impulse *= 0.92  # 8% penalty for 6"
        elif (
            requirements.payload_mass_kg > 20
        ):  # Likely a large rocket even if diameter not specified
            altitude_per_impulse *= 0.92  # 8% drag penalty

        # Calculate impulse needed - use tighter margins to avoid overshooting
        # Calisto: 3350m target would need ~6000 N·s (actual), but we estimate ~6000 N·s
        # So use 1.0-1.1× for optimistic, 0.85-0.9× for pessimistic
        min_impulse_needed = min_alt / (altitude_per_impulse * 1.05)  # Slightly optimistic
        max_impulse_needed = max_alt / (altitude_per_impulse * 0.85)  # Slightly pessimistic

        # Filter motors to likely range (be more inclusive to find working designs)
        viable_motors = [
            m
            for m in self.motors_by_impulse
            if m.total_impulse >= min_impulse_needed * 0.7  # Allow slightly smaller motors
            and m.total_impulse
            <= max_impulse_needed * 1.5  # Allow larger motors (up to O class if needed)
        ]

        # If no viable motors found, expand the range significantly
        if len(viable_motors) == 0:
            log.append(f"⚠ No motors in initial range, expanding search...")
            viable_motors = [
                m
                for m in self.motors_by_impulse
                if m.total_impulse >= min_impulse_needed * 0.5
                and m.total_impulse <= max_impulse_needed * 2.0
            ]

        # Sort by impulse first (smallest to largest), then by cost
        # This ensures we try motors in order of increasing size
        viable_motors.sort(key=lambda m: (m.total_impulse, self._estimate_motor_cost(m)))

        log.append(f"Impulse range: {min_impulse_needed:.0f}-{max_impulse_needed:.0f} N·s")
        log.append(f"Viable motors: {len(viable_motors)}")

        # Adaptive multi-phase search strategy
        # Phase 1: Coarse sampling to find viable range quickly
        # Phase 2: Dense search around successful designs
        # Phase 3: Refinement of best design

        successful_designs = []
        consecutive_failures = 0
        max_consecutive_failures = 15  # Allow more failures before stopping (was 5)

        # Phase 1: Sample motors across the range for initial exploration
        # For heavy rockets, try more motors to find working designs
        if len(viable_motors) > 15:
            sample_count = min(15, len(viable_motors))  # Try more motors (was 8)
            step = max(1, len(viable_motors) // sample_count)
            phase1_motors = [viable_motors[i] for i in range(0, len(viable_motors), step)][
                :sample_count
            ]
            # Also include smallest, largest, and middle
            if viable_motors[0] not in phase1_motors:
                phase1_motors.insert(0, viable_motors[0])
            if viable_motors[-1] not in phase1_motors:
                phase1_motors.append(viable_motors[-1])
            if len(viable_motors) > 2:
                mid_idx = len(viable_motors) // 2
                if viable_motors[mid_idx] not in phase1_motors:
                    phase1_motors.append(viable_motors[mid_idx])
            log.append(f"Phase 1: Coarse search with {len(phase1_motors)} sampled motors")
        else:
            phase1_motors = viable_motors

        phase2_motors = []
        phase2_triggered = False
        tested_motors = set()

        iteration = 0
        for motor in phase1_motors:
            if iteration >= max_iterations:
                break

            # Get appropriate tube for this motor
            # For heavy payloads, we may need a larger tube than minimum motor fit
            # Check if payload requires a larger tube
            min_tube_for_payload = None
            if requirements.payload_mass_kg > 5:
                # For heavy payloads, estimate minimum tube size needed
                # Payload density ~500 kg/m³, need reasonable length (max 2m)
                payload_volume = requirements.payload_mass_kg / 500.0
                min_payload_area = payload_volume / 2.0  # Assume max 2m length
                min_payload_radius = math.sqrt(min_payload_area / math.pi)
                # Find tube that fits payload
                for tube_candidate in self.TUBE_SIZES:
                    if tube_candidate["id"] / 2 >= min_payload_radius:
                        min_tube_for_payload = tube_candidate
                        break

            # Get tube that fits motor
            tube_for_motor = self._get_tube_for_motor(motor.diameter, prefer_smaller=True)

            # Use larger of motor-fit tube or payload-fit tube
            if min_tube_for_payload and min_tube_for_payload["id"] > tube_for_motor["id"]:
                tube = min_tube_for_payload
            else:
                tube = tube_for_motor

            iteration += 1

            # Calculate costs
            body_radius = tube["od"] / 2
            nose_length = body_radius * 4
            # Ensure minimum body length for stability
            min_body_length = max(body_radius * 6, motor.length + 0.5)
            body_length = max(motor.length + 0.6, min_body_length)

            cost = self._calculate_costs(
                tube, body_length, nose_length, motor, requirements.recovery_type
            )

            # Skip if over budget
            if requirements.max_budget_usd and cost.total > requirements.max_budget_usd:
                log.append(
                    f"  [{iteration}] {motor.designation}: Skip (${cost.total:.0f} > budget)"
                )
                continue

            # Skip if already more expensive than best (unless we haven't found any viable designs)
            if best and cost.total >= best_cost:
                continue

            # Run simulation
            altitude, dry_mass, rocket_config, motor_config = self._build_and_simulate(
                tube, motor, requirements
            )

            # Track failures
            if altitude <= 0:
                consecutive_failures += 1
                # Only stop if we've found at least one working design OR too many failures
                if consecutive_failures >= max_consecutive_failures and best is not None:
                    log.append(
                        f"  → Found working design, stopping after {consecutive_failures} failures"
                    )
                    break
                # If no working designs yet, keep trying even with failures
                continue
            consecutive_failures = 0
            tested_motors.add(motor.designation)

            design = DesignPoint(
                body_diameter_m=tube["od"],
                body_length_m=body_length,
                motor_designation=motor.designation,
                motor_impulse_ns=motor.total_impulse,
                simulated_altitude_m=altitude,
                dry_mass_kg=dry_mass,
                cost=cost,
                meets_target=(min_alt <= altitude <= max_alt),
                rocket_config=rocket_config,
                motor_config=motor_config,
            )
            designs.append(design)

            status = "✓" if design.meets_target else "✗"
            log.append(
                f"  {status} [{iteration}] {motor.designation} ({tube['name']}): "
                f"{altitude:.0f}m, ${cost.total:.0f}, {motor.total_impulse:.0f} N·s"
            )

            if callback:
                callback(iteration, max_iterations, design)

            # Update best if this meets target and is cheaper
            if design.meets_target:
                successful_designs.append(design)
                if best is None or cost.total < best_cost:
                    best = design
                    best_cost = cost.total
                    log.append(f"      → New best! ${cost.total:.0f}")

                    # Trigger Phase 2: Dense search around successful designs
                    if not phase2_triggered and len(successful_designs) >= 2:
                        phase2_triggered = True
                        # Find motors in the successful impulse range
                        min_viable_impulse = (
                            min(d.motor_impulse_ns for d in successful_designs) * 0.85
                        )
                        max_viable_impulse = (
                            max(d.motor_impulse_ns for d in successful_designs) * 1.15
                        )
                        phase2_motors = [
                            m
                            for m in viable_motors
                            if min_viable_impulse <= m.total_impulse <= max_viable_impulse
                            and m.designation not in tested_motors
                        ]
                        # Sort by proximity to best design's impulse, then by cost
                        phase2_motors.sort(
                            key=lambda m: (
                                abs(m.total_impulse - best.motor_impulse_ns),
                                self._estimate_motor_cost(m),
                            )
                        )
                        log.append(
                            f"Phase 2: Fine search with {len(phase2_motors)} motors near best (impulse {min_viable_impulse:.0f}-{max_viable_impulse:.0f} N·s)"
                        )

        # Phase 2: Dense search around successful designs
        if phase2_motors and best:
            log.append(f"Phase 2: Refining search around best design")
            for motor in phase2_motors[: max(15, max_iterations - iteration)]:
                if iteration >= max_iterations:
                    break

                # Get appropriate tube (same logic as Phase 1)
                min_tube_for_payload = None
                if requirements.payload_mass_kg > 5:
                    payload_volume = requirements.payload_mass_kg / 500.0
                    min_payload_area = payload_volume / 2.0
                    min_payload_radius = math.sqrt(min_payload_area / math.pi)
                    for tube_candidate in self.TUBE_SIZES:
                        if tube_candidate["id"] / 2 >= min_payload_radius:
                            min_tube_for_payload = tube_candidate
                            break

                tube_for_motor = self._get_tube_for_motor(motor.diameter, prefer_smaller=True)
                if min_tube_for_payload and min_tube_for_payload["id"] > tube_for_motor["id"]:
                    tube = min_tube_for_payload
                else:
                    tube = tube_for_motor
                iteration += 1

                body_radius = tube["od"] / 2
                nose_length = body_radius * 4
                # Ensure minimum body length for stability
                min_body_length = max(body_radius * 6, motor.length + 0.5)
                body_length = max(motor.length + 0.6, min_body_length)

                cost = self._calculate_costs(
                    tube, body_length, nose_length, motor, requirements.recovery_type
                )

                if requirements.max_budget_usd and cost.total > requirements.max_budget_usd:
                    continue

                if best and cost.total >= best_cost * 1.05:  # Allow 5% margin for exploration
                    continue

                altitude, dry_mass, rocket_config, motor_config = self._build_and_simulate(
                    tube, motor, requirements
                )

                if altitude <= 0:
                    continue

                tested_motors.add(motor.designation)

                design = DesignPoint(
                    body_diameter_m=tube["od"],
                    body_length_m=body_length,
                    motor_designation=motor.designation,
                    motor_impulse_ns=motor.total_impulse,
                    simulated_altitude_m=altitude,
                    dry_mass_kg=dry_mass,
                    cost=cost,
                    meets_target=(min_alt <= altitude <= max_alt),
                    rocket_config=rocket_config,
                    motor_config=motor_config,
                )
                designs.append(design)

                status = "✓" if design.meets_target else "✗"
                log.append(
                    f"  {status} [{iteration}] {motor.designation} ({tube['name']}): "
                    f"{altitude:.0f}m, ${cost.total:.0f}, {motor.total_impulse:.0f} N·s"
                )

                if callback:
                    callback(iteration, max_iterations, design)

                if design.meets_target and cost.total < best_cost:
                    best = design
                    best_cost = cost.total
                    log.append(f"      → New best! ${cost.total:.0f}")

        # Summary
        log.append("")
        if best:
            log.append(f"✅ BEST DESIGN: {best.motor_designation}")
            log.append(
                f"   Altitude: {best.simulated_altitude_m:.0f}m (target: {target:.0f}m ± {tol * 100:.0f}%)"
            )
            log.append(f"   Total cost: ${best.cost.total:.0f}")
            log.append(f"   - Motor: ${best.cost.motor_cost:.0f}")
            log.append(f"   - Body: ${best.cost.body_tube_cost:.0f}")
            log.append(
                f"   - Other: ${best.cost.total - best.cost.motor_cost - best.cost.body_tube_cost:.0f}"
            )
            log.append(f"   Body: {best.body_diameter_m * 1000:.0f}mm × {best.body_length_m:.2f}m")
            log.append(f"   Impulse: {best.motor_impulse_ns:.0f} N·s")
            if successful_designs:
                log.append(
                    f"   Tested {len(successful_designs)} successful designs out of {len(designs)} total"
                )
        else:
            log.append("❌ No design found within tolerance and budget")
            if designs:
                # Find closest designs (both above and below target)
                above_target = [d for d in designs if d.simulated_altitude_m > max_alt]
                below_target = [d for d in designs if d.simulated_altitude_m < min_alt]

                if above_target:
                    closest_above = min(above_target, key=lambda d: d.simulated_altitude_m)
                    log.append(
                        f"   Closest above: {closest_above.motor_designation} at {closest_above.simulated_altitude_m:.0f}m (${closest_above.cost.total:.0f})"
                    )
                if below_target:
                    closest_below = max(below_target, key=lambda d: d.simulated_altitude_m)
                    log.append(
                        f"   Closest below: {closest_below.motor_designation} at {closest_below.simulated_altitude_m:.0f}m (${closest_below.cost.total:.0f})"
                    )

                if not above_target and not below_target:
                    closest = min(designs, key=lambda d: abs(d.simulated_altitude_m - target))
                    log.append(
                        f"   Closest: {closest.motor_designation} at {closest.simulated_altitude_m:.0f}m"
                    )

        return best, designs, log

    def optimize_from_text(
        self,
        text: str,
        max_iterations: int = 30,
        callback: Callable = None,
    ) -> Tuple[Optional[DesignPoint], List[DesignPoint], List[str]]:
        """Parse text and optimize."""
        req = self.parse_requirements(text)
        return self.optimize(req, max_iterations, callback)


if __name__ == "__main__":
    from motor_scraper import ThrustCurveScraper

    print("Loading motors...")
    scraper = ThrustCurveScraper()
    motors = scraper.load_motor_database()
    print(f"Loaded {len(motors)} motors")

    optimizer = SmartOptimizer(motors)

    text = "rocket to 5000 ft, as cheap as possible"
    print(f"\n{text}")
    print("=" * 60)

    best, designs, log = optimizer.optimize_from_text(text, max_iterations=20)

    for line in log:
        print(line)
