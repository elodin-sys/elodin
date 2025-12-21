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

        # Parse payload
        payload_patterns = [
            (r"(\d+\.?\d*)\s*(?:lb|lbs)", lambda m: float(m.group(1)) * 0.4536),
            (r"(\d+\.?\d*)\s*kg", lambda m: float(m.group(1))),
        ]
        for pattern, converter in payload_patterns:
            match = re.search(pattern, text_lower)
            if match:
                req.payload_mass_kg = converter(match)
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

    def _get_tube_for_motor(self, motor_diameter: float) -> Dict:
        """Get smallest tube that fits a motor."""
        for tube in self.TUBE_SIZES:
            if tube["id"] >= motor_diameter + 0.003:  # 3mm clearance
                return tube
        return self.TUBE_SIZES[-1]  # Largest

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

        # Calculate dimensions
        body_radius = tube["od"] / 2
        nose_length = body_radius * 4  # 4:1 fineness
        motor_bay_length = motor.length + 0.1
        payload_bay = 0.2 + (requirements.payload_mass_kg * 0.02)  # Scale with payload
        body_length = motor_bay_length + payload_bay + 0.3

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

        # Fins
        fin_span = body_radius * 1.0
        fin_root = body_radius * 1.2
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
        fins.position.x = nose_length + body_length - fin_root
        body.add_child(fins)

        # Motor mount
        mount = InnerTube(
            name="Mount",
            length=motor.length + 0.05,
            outer_radius=motor.diameter / 2 + 0.005,
            thickness=0.003,
        )
        mount.motor_mount = True
        mount.position.x = nose_length + body_length - motor.length - 0.05
        body.add_child(mount)

        # Parachutes
        dry_mass = self._estimate_dry_mass(
            tube,
            body_length,
            nose_length,
            motor,
            requirements.payload_mass_kg,
            requirements.recovery_type,
        )
        landed_mass = dry_mass + motor.case_mass

        # Size for 5 m/s descent
        main_area = (2 * landed_mass * 9.81) / (1.225 * 25 * 1.5)
        main_dia = math.sqrt(4 * main_area / math.pi)

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
            drogue = Parachute(name="Drogue", diameter=drogue_dia, cd=1.3)
            drogue.deployment_event = "APOGEE"
            drogue.deployment_delay = 1.0
            nose.add_child(drogue)

        rocket.calculate_reference_values()

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

        # Run simulation
        try:
            rocket_model = RocketModel(rocket)
            motor_wrapped = Motor.from_openrocket(motor_obj)
            env = Environment()

            solver = FlightSolver(
                rocket=rocket_model,
                motor=motor_wrapped,
                environment=env,
                rail_length=3.0,
                inclination_deg=5.0,
                dt=0.02,
            )
            result = solver.run(max_time=180.0)

            if result.history:
                altitude = max(s.z for s in result.history)
            else:
                altitude = 0.0
        except Exception:
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
        # Typical altitude/impulse ratio: 5-15 m/Ns depending on rocket size
        # Start with conservative (low) estimate to find minimum impulse
        min_impulse_needed = min_alt / 15  # Very light rocket
        max_impulse_needed = max_alt / 3  # Heavier rocket

        # Filter motors to likely range
        viable_motors = [
            m
            for m in self.motors_by_impulse
            if m.total_impulse >= min_impulse_needed * 0.5
            and m.total_impulse <= max_impulse_needed * 2.0
        ]

        # Sort by estimated cost (cheaper motors first for cost optimization)
        viable_motors.sort(key=lambda m: self._estimate_motor_cost(m))

        log.append(f"Impulse range: {min_impulse_needed:.0f}-{max_impulse_needed:.0f} N·s")
        log.append(f"Viable motors: {len(viable_motors)}")

        iteration = 0
        for motor in viable_motors:
            if iteration >= max_iterations:
                break

            # Get appropriate tube for this motor
            tube = self._get_tube_for_motor(motor.diameter)

            iteration += 1

            # Calculate costs
            body_radius = tube["od"] / 2
            nose_length = body_radius * 4
            body_length = motor.length + 0.6

            cost = self._calculate_costs(
                tube, body_length, nose_length, motor, requirements.recovery_type
            )

            # Skip if over budget
            if requirements.max_budget_usd and cost.total > requirements.max_budget_usd:
                log.append(
                    f"  [{iteration}] {motor.designation}: Skip (${cost.total:.0f} > budget)"
                )
                continue

            # Skip if already more expensive than best
            if cost.total >= best_cost:
                continue

            # Run simulation
            altitude, dry_mass, rocket_config, motor_config = self._build_and_simulate(
                tube, motor, requirements
            )

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
                f"{altitude:.0f}m, ${cost.total:.0f}"
            )

            if callback:
                callback(iteration, max_iterations, design)

            # Update best if this meets target and is cheaper
            if design.meets_target and cost.total < best_cost:
                best = design
                best_cost = cost.total
                log.append(f"      → New best! ${cost.total:.0f}")

        # Summary
        log.append("")
        if best:
            log.append(f"✅ BEST DESIGN: {best.motor_designation}")
            log.append(f"   Altitude: {best.simulated_altitude_m:.0f}m")
            log.append(f"   Total cost: ${best.cost.total:.0f}")
            log.append(f"   - Motor: ${best.cost.motor_cost:.0f}")
            log.append(f"   - Body: ${best.cost.body_tube_cost:.0f}")
            log.append(
                f"   - Other: ${best.cost.total - best.cost.motor_cost - best.cost.body_tube_cost:.0f}"
            )
        else:
            log.append("❌ No design found within tolerance and budget")
            if designs:
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
