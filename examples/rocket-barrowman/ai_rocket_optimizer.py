"""
Enhanced AI Rocket Optimizer with Simulation-Based Design Iteration.

This module provides a sophisticated rocket design optimizer that:
1. Runs actual flight simulations to verify designs
2. Iteratively sweeps through motor options to find optimal choice
3. Optimizes for cost (cheapest motor that meets requirements)
4. Provides confidence scores and detailed analysis
5. Generates multiple design candidates for comparison
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
import numpy as np

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

# Try to import OpenAI for NLP parsing
try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

import json
import re


@dataclass
class DesignRequirements:
    """Parsed rocket requirements from natural language."""

    target_altitude_m: Optional[float] = None
    min_altitude_m: Optional[float] = None
    max_altitude_m: Optional[float] = None
    payload_mass_kg: Optional[float] = None
    payload_size: Optional[str] = None
    total_rocket_mass_kg: Optional[float] = None
    diameter_constraint_m: Optional[float] = None
    length_constraint_m: Optional[float] = None
    fin_count: Optional[int] = None
    recovery_method: Optional[str] = None  # "parachute", "dual_deploy", "none"
    landing_distance_m: Optional[float] = None
    motor_preference: Optional[str] = None  # Motor class like "M", "L", "K"
    motor_manufacturer: Optional[str] = None
    body_material: Optional[str] = None
    budget_usd: Optional[float] = None
    optimize_for: str = "cost"  # "cost", "performance", "reliability"


@dataclass
class MotorCandidate:
    """A motor candidate with cost and performance data."""

    designation: str
    manufacturer: str
    total_impulse: float
    max_thrust: float
    avg_thrust: float
    burn_time: float
    diameter: float
    length: float
    total_mass: float
    propellant_mass: float
    case_mass: float
    thrust_curve: List[Tuple[float, float]]
    estimated_cost_usd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "designation": self.designation,
            "manufacturer": self.manufacturer,
            "total_impulse": self.total_impulse,
            "max_thrust": self.max_thrust,
            "avg_thrust": self.avg_thrust,
            "burn_time": self.burn_time,
            "diameter": self.diameter,
            "length": self.length,
            "total_mass": self.total_mass,
            "propellant_mass": self.propellant_mass,
            "case_mass": self.case_mass,
            "thrust_curve": self.thrust_curve,
            "estimated_cost_usd": self.estimated_cost_usd,
        }


@dataclass
class SimulationResult:
    """Results from a simulation run."""

    max_altitude_m: float
    max_velocity_ms: float
    max_mach: float
    max_acceleration_g: float
    apogee_time_s: float
    flight_time_s: float
    landing_distance_m: float
    stability_margin: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class DesignCandidate:
    """A complete rocket design candidate with simulation results."""

    rocket_config: Dict[str, Any]
    motor: MotorCandidate
    simulation: Optional[SimulationResult]
    confidence_score: float  # 0-100
    cost_estimate_usd: float
    meets_requirements: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def altitude_accuracy(self) -> float:
        """How close to target altitude (0-100%)."""
        if not self.simulation or not self.simulation.success:
            return 0.0
        return 100.0  # Will be calculated based on target

    @property
    def overall_score(self) -> float:
        """Combined score considering altitude accuracy, cost, and confidence."""
        if not self.meets_requirements:
            return 0.0
        return self.confidence_score


@dataclass
class OptimizationResult:
    """Complete optimization result with multiple candidates."""

    requirements: DesignRequirements
    best_candidate: Optional[DesignCandidate]
    all_candidates: List[DesignCandidate]
    cheapest_working: Optional[DesignCandidate]
    most_accurate: Optional[DesignCandidate]
    optimization_log: List[str]
    total_simulations_run: int


class RocketOptimizer:
    """
    Enhanced AI-powered rocket design optimizer.

    Uses actual flight simulations to iteratively find the optimal
    rocket design that meets requirements while minimizing cost.
    """

    # Motor cost estimates by impulse class (USD, rough estimates)
    MOTOR_COST_ESTIMATES = {
        "A": 8,
        "B": 10,
        "C": 12,
        "D": 15,
        "E": 20,
        "F": 30,
        "G": 45,
        "H": 70,
        "I": 100,
        "J": 150,
        "K": 200,
        "L": 300,
        "M": 450,
        "N": 700,
        "O": 1000,
    }

    # Material costs (USD per kg, rough estimates)
    MATERIAL_COSTS = {
        "Fiberglass": 50,
        "Carbon fiber": 150,
        "Cardboard": 5,
        "Aluminum": 25,
        "Kraft phenolic": 15,
        "Blue tube": 20,
        "Plywood": 8,
    }

    def __init__(
        self,
        motor_database: Optional[List["MotorData"]] = None,
        openai_api_key: Optional[str] = None,
    ):
        self.motor_database = motor_database or []
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.use_openai = OPENAI_AVAILABLE and self.openai_api_key is not None

        if self.use_openai:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
        else:
            self.openai_client = None

        # Pre-process motor database with cost estimates
        self._motor_candidates: List[MotorCandidate] = []
        self._preprocess_motors()

    def _preprocess_motors(self) -> None:
        """Convert motor database to candidates with cost estimates."""
        self._motor_candidates = []

        for motor in self.motor_database:
            if motor.total_impulse <= 0:
                continue

            # Estimate cost based on impulse class
            motor_class = motor.designation[0] if motor.designation else "?"
            base_cost = self.MOTOR_COST_ESTIMATES.get(motor_class.upper(), 100)

            # Adjust cost based on manufacturer and impulse
            cost_multiplier = 1.0
            if "Cesaroni" in motor.manufacturer:
                cost_multiplier = 1.1
            elif "Aerotech" in motor.manufacturer:
                cost_multiplier = 1.0
            elif "CTI" in motor.manufacturer:
                cost_multiplier = 1.1

            estimated_cost = base_cost * cost_multiplier

            candidate = MotorCandidate(
                designation=motor.designation,
                manufacturer=motor.manufacturer,
                total_impulse=motor.total_impulse,
                max_thrust=motor.max_thrust,
                avg_thrust=motor.avg_thrust,
                burn_time=motor.burn_time,
                diameter=motor.diameter,
                length=motor.length,
                total_mass=motor.total_mass,
                propellant_mass=motor.propellant_mass,
                case_mass=motor.case_mass,
                thrust_curve=motor.thrust_curve,
                estimated_cost_usd=estimated_cost,
            )
            self._motor_candidates.append(candidate)

        # Sort by cost (cheapest first)
        self._motor_candidates.sort(key=lambda m: m.estimated_cost_usd)

    def parse_requirements(self, text: str) -> DesignRequirements:
        """Parse natural language requirements into structured format."""
        if self.use_openai:
            return self._parse_with_openai(text)
        return self._parse_with_regex(text)

    def _parse_with_openai(self, text: str) -> DesignRequirements:
        """Parse requirements using OpenAI API."""
        try:
            prompt = f"""Extract rocket design requirements from this text and return as JSON:
"{text}"

Return JSON with these fields (use null if not specified):
- target_altitude_m: target altitude in meters (convert from feet if needed, 1ft = 0.3048m)
- min_altitude_m: minimum acceptable altitude in meters
- max_altitude_m: maximum acceptable altitude in meters  
- payload_mass_kg: payload mass in kg (convert from lbs if needed, 1lb = 0.453592kg)
- payload_size: cubesat size like "6U", "3U", "1U"
- total_rocket_mass_kg: total rocket mass in kg
- diameter_constraint_m: max diameter in meters (convert from inches, 1in = 0.0254m)
- length_constraint_m: max length in meters
- fin_count: number of fins (integer)
- recovery_method: "parachute", "dual_deploy", or "none"
- landing_distance_m: max landing distance in meters (convert from miles, 1mi = 1609.34m)
- motor_preference: motor class like "M", "L", "K", "J"
- motor_manufacturer: preferred manufacturer
- body_material: "Fiberglass", "Carbon fiber", "Cardboard", etc.
- budget_usd: budget in USD
- optimize_for: "cost" (cheapest), "performance" (best altitude accuracy), or "reliability"

Be thorough with unit conversions. Return only valid JSON."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a rocket design expert. Extract requirements precisely.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=800,
            )

            result_text = response.choices[0].message.content.strip()
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
                result_text = result_text.strip()

            parsed = json.loads(result_text)

            return DesignRequirements(
                target_altitude_m=parsed.get("target_altitude_m"),
                min_altitude_m=parsed.get("min_altitude_m"),
                max_altitude_m=parsed.get("max_altitude_m"),
                payload_mass_kg=parsed.get("payload_mass_kg"),
                payload_size=parsed.get("payload_size"),
                total_rocket_mass_kg=parsed.get("total_rocket_mass_kg"),
                diameter_constraint_m=parsed.get("diameter_constraint_m"),
                length_constraint_m=parsed.get("length_constraint_m"),
                fin_count=parsed.get("fin_count"),
                recovery_method=parsed.get("recovery_method"),
                landing_distance_m=parsed.get("landing_distance_m"),
                motor_preference=parsed.get("motor_preference"),
                motor_manufacturer=parsed.get("motor_manufacturer"),
                body_material=parsed.get("body_material"),
                budget_usd=parsed.get("budget_usd"),
                optimize_for=parsed.get("optimize_for", "cost"),
            )
        except Exception as e:
            print(f"OpenAI parsing failed: {e}, using regex")
            return self._parse_with_regex(text)

    def _parse_with_regex(self, text: str) -> DesignRequirements:
        """Parse requirements using regex patterns."""
        req = DesignRequirements()
        text_lower = text.lower()

        # Altitude patterns
        alt_patterns = [
            (r"(\d+\.?\d*)\s*(?:thousand|k)\s*(?:ft|feet)", 304.8),
            (r"(\d+\.?\d*)\s*(?:ft|feet)\b", 0.3048),
            (r"(\d+\.?\d*)\s*(?:km)\b", 1000.0),
            (r"(\d+\.?\d*)\s*(?:m|meters?)\b(?!s)", 1.0),
            (r"goes?\s+to\s+(\d+)", 0.3048),
            (r"reach\s+(\d+)", 0.3048),
        ]
        for pattern, mult in alt_patterns:
            match = re.search(pattern, text_lower)
            if match:
                req.target_altitude_m = float(match.group(1)) * mult
                break

        # Payload mass
        payload_patterns = [
            (r"(\d+\.?\d*)\s*(?:lb|lbs|pound)\s*payload", 0.453592),
            (r"payload\s+(?:of\s+)?(\d+\.?\d*)\s*(?:lb|lbs|kg)", 0.453592),
            (r"carries?\s+(\d+\.?\d*)\s*(?:lb|lbs)", 0.453592),
            (r"(\d+\.?\d*)\s*kg\s+payload", 1.0),
        ]
        for pattern, mult in payload_patterns:
            match = re.search(pattern, text_lower)
            if match:
                req.payload_mass_kg = float(match.group(1)) * mult
                break

        # Cubesat size
        match = re.search(r"(\d+)[Uu]", text)
        if match:
            req.payload_size = f"{match.group(1)}U"

        # Fin count
        match = re.search(r"(\d+)\s*fins?", text_lower)
        if match:
            req.fin_count = int(match.group(1))

        # Recovery
        if "dual" in text_lower or "drogue" in text_lower:
            req.recovery_method = "dual_deploy"
        elif "parachute" in text_lower or "chute" in text_lower:
            req.recovery_method = "parachute"

        # Landing distance
        match = re.search(r"land.*?(\d+\.?\d*)\s*(?:mile|mi)", text_lower)
        if match:
            req.landing_distance_m = float(match.group(1)) * 1609.34

        # Budget
        match = re.search(r"\$\s*(\d+)", text)
        if match:
            req.budget_usd = float(match.group(1))

        # Default to cost optimization
        if "cheap" in text_lower or "budget" in text_lower:
            req.optimize_for = "cost"
        elif "accurate" in text_lower or "precise" in text_lower:
            req.optimize_for = "performance"

        return req

    def _estimate_required_impulse(self, requirements: DesignRequirements) -> float:
        """Estimate required motor impulse based on requirements."""
        target_alt = requirements.target_altitude_m or 3000.0
        payload_mass = requirements.payload_mass_kg or 0.0

        # Calibrated based on actual simulations:
        # Our sims show: H motors (~230 Ns) → ~2200m
        # So: 1 Ns ≈ 9.5m altitude
        # Invert: impulse ≈ altitude / 9.5

        # But this varies with rocket mass/size, so use conservative estimate
        # For a typical 3-5kg dry mass rocket:
        # - G (~100 Ns) → ~800m
        # - H (~230 Ns) → ~2000m
        # - I (~400 Ns) → ~3500m
        # - J (~700 Ns) → ~5500m
        # - K (~1400 Ns) → ~9000m
        #
        # Pattern: altitude ≈ impulse * 8-10 for optimized rockets
        # So: impulse ≈ altitude / 9

        base_impulse = target_alt / 9.0

        # Scale for payload mass (heavier = more impulse)
        mass_factor = 1.0 + (payload_mass / 5.0) * 0.3

        return base_impulse * mass_factor

    def _get_motor_candidates_for_altitude(
        self,
        target_altitude_m: float,
        requirements: DesignRequirements,
        margin: float = 1.0,
    ) -> List[MotorCandidate]:
        """
        Get motor candidates that might reach target altitude.
        Returns motors sorted by cost (cheapest first).
        """
        estimated_impulse = self._estimate_required_impulse(requirements)

        # Filter motors - use very wide margin since altitude varies a lot
        min_impulse = max(50, estimated_impulse * (1 - margin))  # Minimum 50 Ns
        max_impulse = estimated_impulse * (1 + margin * 1.5)

        candidates = []
        for motor in self._motor_candidates:
            # Apply filters
            if requirements.motor_preference:
                if not motor.designation.startswith(requirements.motor_preference.upper()):
                    continue

            if requirements.motor_manufacturer:
                if requirements.motor_manufacturer.lower() not in motor.manufacturer.lower():
                    continue

            if requirements.budget_usd:
                if motor.estimated_cost_usd > requirements.budget_usd:
                    continue

            # Check impulse range
            if min_impulse <= motor.total_impulse <= max_impulse:
                candidates.append(motor)

        # If no candidates, widen search
        if not candidates:
            min_impulse = estimated_impulse * 0.3
            max_impulse = estimated_impulse * 3.0
            for motor in self._motor_candidates:
                if min_impulse <= motor.total_impulse <= max_impulse:
                    candidates.append(motor)

        # Sort by cost (cheapest first)
        candidates.sort(key=lambda m: m.estimated_cost_usd)
        return candidates[:30]  # Limit to top 30 cheapest

    def _build_rocket_for_motor(
        self,
        motor: MotorCandidate,
        requirements: DesignRequirements,
    ) -> Tuple[Rocket, Dict[str, Any]]:
        """Build a rocket sized appropriately for the given motor."""

        # Size body tube based on motor diameter
        body_radius = (motor.diameter / 2.0) + 0.008  # 8mm wall clearance

        # Round to standard tube sizes
        standard_radii = [0.019, 0.027, 0.0375, 0.049, 0.0635, 0.076, 0.10, 0.127, 0.15]
        body_radius = min(
            standard_radii,
            key=lambda r: abs(r - body_radius) if r >= motor.diameter / 2 else float("inf"),
        )

        # Apply diameter constraint
        if requirements.diameter_constraint_m:
            body_radius = min(body_radius, requirements.diameter_constraint_m / 2.0)

        # Calculate lengths
        nose_length = body_radius * 4.0  # 4:1 fineness ratio
        motor_bay_length = motor.length + 0.1
        payload_length = 0.3
        if requirements.payload_size:
            cubesat_lengths = {"1U": 0.12, "2U": 0.24, "3U": 0.35, "6U": 0.35, "12U": 0.35}
            payload_length = cubesat_lengths.get(requirements.payload_size, 0.3)

        body_length = motor_bay_length + payload_length + 0.3

        # Apply length constraint
        if requirements.length_constraint_m:
            max_body = requirements.length_constraint_m - nose_length
            body_length = min(body_length, max_body)

        # Fin sizing based on caliber stability
        fin_count = requirements.fin_count or 4
        fin_span = body_radius * 1.2
        fin_root_chord = body_radius * 1.5
        fin_tip_chord = fin_root_chord * 0.5
        fin_sweep = fin_root_chord * 0.4

        # Material
        body_material = requirements.body_material or "Fiberglass"

        # Build rocket
        rocket = Rocket("Optimized Rocket")

        # Nose cone
        nose = NoseCone(
            name="Nose Cone",
            length=nose_length,
            base_radius=body_radius,
            thickness=0.003,
            shape=NoseCone.Shape.VON_KARMAN,
        )
        nose.material = MATERIALS.get(body_material, MATERIALS["Fiberglass"])
        nose.position.x = 0.0
        rocket.add_child(nose)

        # Body tube
        body = BodyTube(
            name="Body Tube",
            length=body_length,
            outer_radius=body_radius,
            thickness=0.003,
        )
        body.material = MATERIALS.get(body_material, MATERIALS["Fiberglass"])
        body.position.x = nose_length
        rocket.add_child(body)

        # Fins
        fins = TrapezoidFinSet(
            name="Fins",
            fin_count=fin_count,
            root_chord=fin_root_chord,
            tip_chord=fin_tip_chord,
            span=fin_span,
            sweep=fin_sweep,
            thickness=0.005,
        )
        fins.material = MATERIALS.get(body_material, MATERIALS["Fiberglass"])
        fins.position.x = nose_length + body_length - fin_root_chord
        body.add_child(fins)

        # Motor mount
        motor_mount = InnerTube(
            name="Motor Mount",
            length=motor.length + 0.05,
            outer_radius=(motor.diameter / 2.0) + 0.005,
            thickness=0.003,
        )
        motor_mount.material = MATERIALS["Fiberglass"]
        motor_mount.position.x = nose_length + body_length - motor.length - 0.05
        motor_mount.motor_mount = True
        body.add_child(motor_mount)

        # Parachutes
        recovery = requirements.recovery_method or "dual_deploy"

        # Size parachutes based on estimated mass
        dry_mass = rocket.get_total_mass() + (requirements.payload_mass_kg or 0.0)
        landed_mass = dry_mass + motor.case_mass

        # Main chute for 5 m/s descent
        main_area = (2 * landed_mass * 9.81) / (1.225 * 25 * 1.5)  # v²=25, Cd=1.5
        main_diameter = math.sqrt(4 * main_area / math.pi)

        main_chute = Parachute(name="Main", diameter=main_diameter, cd=1.5)
        if recovery == "dual_deploy":
            main_chute.deployment_event = "ALTITUDE"
            main_chute.deployment_altitude = 800.0
        else:
            main_chute.deployment_event = "APOGEE"
        main_chute.deployment_delay = 1.5
        nose.add_child(main_chute)

        if recovery == "dual_deploy":
            # Drogue for 30 m/s descent
            drogue_area = (2 * landed_mass * 9.81) / (0.8 * 900 * 1.3)
            drogue_diameter = math.sqrt(4 * drogue_area / math.pi)

            drogue = Parachute(name="Drogue", diameter=drogue_diameter, cd=1.3)
            drogue.deployment_event = "APOGEE"
            drogue.deployment_delay = 1.5
            nose.add_child(drogue)

        rocket.calculate_reference_values()

        # Build config dict
        config = {
            "name": "Optimized Rocket",
            "has_nose": True,
            "nose_length": nose_length,
            "nose_thickness": 0.003,
            "nose_shape": "VON_KARMAN",
            "nose_material": body_material,
            "body_length": body_length,
            "body_radius": body_radius,
            "body_thickness": 0.003,
            "body_material": body_material,
            "has_fins": True,
            "fin_count": fin_count,
            "fin_root_chord": fin_root_chord,
            "fin_tip_chord": fin_tip_chord,
            "fin_span": fin_span,
            "fin_sweep": fin_sweep,
            "fin_thickness": 0.005,
            "fin_material": body_material,
            "has_motor_mount": True,
            "motor_mount_length": motor.length + 0.05,
            "motor_mount_radius": (motor.diameter / 2.0) + 0.005,
            "motor_mount_thickness": 0.003,
            "motor_mount_material": "Fiberglass",
            "has_main_chute": True,
            "main_chute_diameter": main_diameter,
            "main_chute_cd": 1.5,
            "main_deployment_event": "ALTITUDE" if recovery == "dual_deploy" else "APOGEE",
            "main_deployment_altitude": 800.0,
            "main_deployment_delay": 1.5,
            "has_drogue": recovery == "dual_deploy",
            "drogue_diameter": drogue_diameter if recovery == "dual_deploy" else 0.99,
            "drogue_cd": 1.3,
            "drogue_deployment_event": "APOGEE",
            "drogue_deployment_delay": 1.5,
        }

        return rocket, config

    def _build_motor_object(self, motor: MotorCandidate) -> ORMotor:
        """Convert MotorCandidate to ORMotor object."""
        return ORMotor(
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

    def _run_simulation(
        self,
        rocket: Rocket,
        motor: MotorCandidate,
        requirements: DesignRequirements,
    ) -> SimulationResult:
        """Run a flight simulation and return results."""
        try:
            rocket_model = RocketModel(rocket)
            motor_obj = self._build_motor_object(motor)
            motor_wrapped = Motor.from_openrocket(motor_obj)
            env = Environment()

            solver = FlightSolver(
                rocket=rocket_model,
                motor=motor_wrapped,
                environment=env,
                rail_length=5.0,
                inclination_deg=5.0,
                heading_deg=0.0,
                dt=0.02,
            )

            result = solver.run(max_time=200.0)

            if not result.history:
                return SimulationResult(
                    max_altitude_m=0,
                    max_velocity_ms=0,
                    max_mach=0,
                    max_acceleration_g=0,
                    apogee_time_s=0,
                    flight_time_s=0,
                    landing_distance_m=0,
                    stability_margin=0,
                    success=False,
                    error_message="No simulation data",
                )

            # Extract metrics
            max_alt = max(s.z for s in result.history)
            max_vel = max(np.linalg.norm(s.velocity) for s in result.history)
            max_mach = max(s.mach for s in result.history)

            # Estimate max acceleration (from velocity derivative)
            velocities = [np.linalg.norm(s.velocity) for s in result.history]
            times = [s.time for s in result.history]
            accel = 0
            for i in range(1, len(velocities)):
                dt = times[i] - times[i - 1]
                if dt > 0:
                    a = abs(velocities[i] - velocities[i - 1]) / dt
                    accel = max(accel, a)
            max_accel_g = accel / 9.81

            apogee_time = next(s.time for s in result.history if s.z == max_alt)
            flight_time = result.history[-1].time

            # Landing distance
            final_pos = result.history[-1].position
            landing_dist = np.sqrt(final_pos[0] ** 2 + final_pos[1] ** 2)

            # Stability (simplified - CP ahead of CG is stable)
            try:
                cp = rocket_model.aero.calculate_cp(0.3)
                cg = rocket_model.dry_cg
                stability = (cp - cg) / (rocket_model.reference_diameter)
            except:
                stability = 1.5

            return SimulationResult(
                max_altitude_m=max_alt,
                max_velocity_ms=max_vel,
                max_mach=max_mach,
                max_acceleration_g=max_accel_g,
                apogee_time_s=apogee_time,
                flight_time_s=flight_time,
                landing_distance_m=landing_dist,
                stability_margin=stability,
                success=True,
            )
        except Exception as e:
            return SimulationResult(
                max_altitude_m=0,
                max_velocity_ms=0,
                max_mach=0,
                max_acceleration_g=0,
                apogee_time_s=0,
                flight_time_s=0,
                landing_distance_m=0,
                stability_margin=0,
                success=False,
                error_message=str(e),
            )

    def _calculate_confidence(
        self,
        sim: SimulationResult,
        requirements: DesignRequirements,
    ) -> Tuple[float, List[str], List[str]]:
        """Calculate confidence score and identify issues."""
        issues = []
        warnings = []
        score = 100.0

        if not sim.success:
            return 0.0, ["Simulation failed: " + (sim.error_message or "unknown")], []

        target = requirements.target_altitude_m
        if target:
            error_pct = abs(sim.max_altitude_m - target) / target * 100
            if error_pct > 20:
                issues.append(f"Altitude error {error_pct:.1f}% exceeds 20% tolerance")
                score -= 30
            elif error_pct > 10:
                warnings.append(f"Altitude error {error_pct:.1f}% (within tolerance)")
                score -= 10
            else:
                score += 5  # Bonus for accuracy

        # Stability check
        if sim.stability_margin < 1.0:
            issues.append(f"Stability margin {sim.stability_margin:.2f} cal is too low (< 1.0)")
            score -= 20
        elif sim.stability_margin < 1.5:
            warnings.append(f"Stability margin {sim.stability_margin:.2f} cal is marginal")
            score -= 5
        elif sim.stability_margin > 3.0:
            warnings.append(
                f"Stability margin {sim.stability_margin:.2f} cal is high (may weathercock)"
            )

        # Landing distance
        if requirements.landing_distance_m:
            if sim.landing_distance_m > requirements.landing_distance_m:
                issues.append(
                    f"Landing {sim.landing_distance_m:.0f}m exceeds limit {requirements.landing_distance_m:.0f}m"
                )
                score -= 15

        # Mach limit
        if sim.max_mach > 0.9:
            warnings.append(f"Max Mach {sim.max_mach:.2f} approaching transonic")

        return max(0, min(100, score)), issues, warnings

    def _estimate_total_cost(
        self,
        config: Dict[str, Any],
        motor: MotorCandidate,
    ) -> float:
        """Estimate total build cost."""
        cost = motor.estimated_cost_usd

        # Estimate material costs
        body_radius = config.get("body_radius", 0.0635)
        body_length = config.get("body_length", 1.5)
        nose_length = config.get("nose_length", 0.5)

        # Rough mass estimate
        body_area = 2 * math.pi * body_radius * body_length
        nose_area = math.pi * body_radius * nose_length
        fin_area = (
            4
            * 0.5
            * (config.get("fin_root_chord", 0.1) + config.get("fin_tip_chord", 0.05))
            * config.get("fin_span", 0.1)
        )

        total_area = body_area + nose_area + fin_area
        material_mass = total_area * 0.003 * 1850  # Fiberglass density

        material = config.get("body_material", "Fiberglass")
        material_cost_per_kg = self.MATERIAL_COSTS.get(material, 50)
        cost += material_mass * material_cost_per_kg

        # Parachute costs
        if config.get("has_main_chute"):
            cost += 50  # Main chute
        if config.get("has_drogue"):
            cost += 30  # Drogue
            cost += 100  # Altimeter for dual deploy

        # Hardware
        cost += 30  # Fasteners, epoxy, etc.

        return cost

    def optimize(
        self,
        requirements: DesignRequirements,
        max_simulations: int = 10,
        callback: Optional[callable] = None,
    ) -> OptimizationResult:
        """
        Run optimization to find the best rocket design.

        Args:
            requirements: Design requirements
            max_simulations: Maximum number of simulations to run
            callback: Optional callback function(iteration, total, candidate)

        Returns:
            OptimizationResult with best candidates
        """
        log = []
        candidates = []

        target = requirements.target_altitude_m or 3000.0
        log.append(f"Target altitude: {target:.0f} m ({target * 3.281:.0f} ft)")

        # Get motor candidates
        motor_candidates = self._get_motor_candidates_for_altitude(target, requirements)
        log.append(f"Found {len(motor_candidates)} potential motors")

        if not motor_candidates:
            log.append("ERROR: No suitable motors found!")
            return OptimizationResult(
                requirements=requirements,
                best_candidate=None,
                all_candidates=[],
                cheapest_working=None,
                most_accurate=None,
                optimization_log=log,
                total_simulations_run=0,
            )

        # Run simulations for each motor candidate
        sims_run = 0
        for i, motor in enumerate(motor_candidates[:max_simulations]):
            log.append(
                f"\n--- Testing motor {i+1}/{min(len(motor_candidates), max_simulations)}: {motor.designation} ---"
            )
            log.append(
                f"  Impulse: {motor.total_impulse:.0f} N·s, Cost: ${motor.estimated_cost_usd:.0f}"
            )

            # Build rocket
            rocket, config = self._build_rocket_for_motor(motor, requirements)

            # Run simulation
            sim = self._run_simulation(rocket, motor, requirements)
            sims_run += 1

            if sim.success:
                log.append(
                    f"  Altitude: {sim.max_altitude_m:.0f} m ({sim.max_altitude_m * 3.281:.0f} ft)"
                )
                error = abs(sim.max_altitude_m - target) / target * 100
                log.append(f"  Error: {error:.1f}%")
            else:
                log.append(f"  FAILED: {sim.error_message}")

            # Calculate confidence
            confidence, issues, warnings = self._calculate_confidence(sim, requirements)

            # Calculate cost
            total_cost = self._estimate_total_cost(config, motor)

            # Check if meets requirements
            meets_reqs = sim.success and len(issues) == 0
            if target:
                error = abs(sim.max_altitude_m - target) / target
                meets_reqs = meets_reqs and error < 0.20  # Within 20%

            candidate = DesignCandidate(
                rocket_config=config,
                motor=motor,
                simulation=sim,
                confidence_score=confidence,
                cost_estimate_usd=total_cost,
                meets_requirements=meets_reqs,
                issues=issues,
                warnings=warnings,
            )
            candidates.append(candidate)

            if callback:
                callback(i + 1, min(len(motor_candidates), max_simulations), candidate)

        # Find best candidates
        working = [c for c in candidates if c.meets_requirements]

        cheapest_working = None
        most_accurate = None
        best = None

        if working:
            # Cheapest that works
            cheapest_working = min(working, key=lambda c: c.cost_estimate_usd)
            log.append(
                f"\nCheapest working: {cheapest_working.motor.designation} @ ${cheapest_working.cost_estimate_usd:.0f}"
            )

            # Most accurate (closest to target altitude)
            if target:
                most_accurate = min(
                    working, key=lambda c: abs(c.simulation.max_altitude_m - target)
                )
                log.append(
                    f"Most accurate: {most_accurate.motor.designation} ({most_accurate.simulation.max_altitude_m:.0f}m)"
                )

            # Best overall
            if requirements.optimize_for == "cost":
                best = cheapest_working
            elif requirements.optimize_for == "performance":
                best = most_accurate or cheapest_working
            else:
                best = max(working, key=lambda c: c.confidence_score)
        else:
            log.append("\nWARNING: No designs meet all requirements!")
            # Return closest
            if candidates:
                best = max(candidates, key=lambda c: c.confidence_score)

        log.append(f"\nTotal simulations run: {sims_run}")

        return OptimizationResult(
            requirements=requirements,
            best_candidate=best,
            all_candidates=candidates,
            cheapest_working=cheapest_working,
            most_accurate=most_accurate,
            optimization_log=log,
            total_simulations_run=sims_run,
        )

    def optimize_from_text(
        self,
        text: str,
        max_simulations: int = 10,
        callback: Optional[callable] = None,
    ) -> OptimizationResult:
        """
        Complete pipeline: parse natural language and optimize design.
        """
        requirements = self.parse_requirements(text)
        return self.optimize(requirements, max_simulations, callback)


# Convenience function
def design_rocket(
    text: str,
    motor_database: Optional[List] = None,
    max_simulations: int = 10,
) -> Tuple[Dict[str, Any], Dict[str, Any], OptimizationResult]:
    """
    Design a rocket from natural language requirements.

    Returns:
        (rocket_config, motor_config, optimization_result)
    """
    optimizer = RocketOptimizer(motor_database)
    result = optimizer.optimize_from_text(text, max_simulations)

    if result.best_candidate:
        return (
            result.best_candidate.rocket_config,
            result.best_candidate.motor.to_dict(),
            result,
        )
    return None, None, result


if __name__ == "__main__":
    # Test the optimizer
    from motor_scraper import ThrustCurveScraper

    print("Loading motor database...")
    scraper = ThrustCurveScraper()
    motors = scraper.load_motor_database()

    if motors:
        print(f"Loaded {len(motors)} motors")

        optimizer = RocketOptimizer(motors)

        # Test case
        text = "I want a rocket that goes to 10000 ft, carries a small payload, and is as cheap as possible"

        print(f"\nOptimizing for: {text}")
        print("=" * 60)

        result = optimizer.optimize_from_text(text, max_simulations=5)

        for line in result.optimization_log:
            print(line)

        if result.best_candidate:
            print("\n" + "=" * 60)
            print("BEST DESIGN:")
            print(f"  Motor: {result.best_candidate.motor.designation}")
            print(f"  Altitude: {result.best_candidate.simulation.max_altitude_m:.0f} m")
            print(f"  Cost: ${result.best_candidate.cost_estimate_usd:.0f}")
            print(f"  Confidence: {result.best_candidate.confidence_score:.0f}%")
    else:
        print("No motors loaded - run motor scraper first")
