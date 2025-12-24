"""
Structural Analysis Module for Rocket Flight Simulation.

Provides:
- Fin flutter analysis (NACA TN-4197 method)
- Structural loads (axial, bending, shear)
- Stress analysis
- Safety margins
- Material failure prediction
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum


class LoadType(Enum):
    """Types of structural loads"""

    AXIAL = "axial"  # Thrust, drag, inertia
    BENDING = "bending"  # Aerodynamic moments
    SHEAR = "shear"  # Lateral loads
    TORSION = "torsion"  # Roll moments
    PRESSURE = "pressure"  # Internal/external pressure


@dataclass
class MaterialProperties:
    """Material mechanical properties for structural analysis"""

    name: str
    density: float  # kg/m³
    youngs_modulus: float  # Pa (E)
    shear_modulus: float  # Pa (G)
    yield_strength: float  # Pa
    ultimate_strength: float  # Pa
    poissons_ratio: float  # ν
    thermal_expansion: float  # 1/K

    @property
    def bulk_modulus(self) -> float:
        """Bulk modulus K = E / (3(1-2ν))"""
        return self.youngs_modulus / (3 * (1 - 2 * self.poissons_ratio))


# Standard aerospace materials
STRUCTURAL_MATERIALS = {
    "Fiberglass": MaterialProperties(
        name="Fiberglass (E-glass)",
        density=1850,
        youngs_modulus=35e9,
        shear_modulus=14e9,
        yield_strength=200e6,
        ultimate_strength=350e6,
        poissons_ratio=0.25,
        thermal_expansion=6e-6,
    ),
    "Carbon fiber": MaterialProperties(
        name="Carbon Fiber (T300)",
        density=1780,
        youngs_modulus=230e9,
        shear_modulus=90e9,
        yield_strength=600e6,
        ultimate_strength=1500e6,
        poissons_ratio=0.28,
        thermal_expansion=0.5e-6,
    ),
    "Aluminum": MaterialProperties(
        name="Aluminum 6061-T6",
        density=2700,
        youngs_modulus=69e9,
        shear_modulus=26e9,
        yield_strength=276e6,
        ultimate_strength=310e6,
        poissons_ratio=0.33,
        thermal_expansion=23e-6,
    ),
    "Plywood (birch)": MaterialProperties(
        name="Baltic Birch Plywood",
        density=630,
        youngs_modulus=12e9,
        shear_modulus=0.7e9,
        yield_strength=40e6,
        ultimate_strength=70e6,
        poissons_ratio=0.3,
        thermal_expansion=5e-6,
    ),
    "Blue tube": MaterialProperties(
        name="Blue Tube (phenolic)",
        density=1040,
        youngs_modulus=10e9,
        shear_modulus=4e9,
        yield_strength=80e6,
        ultimate_strength=120e6,
        poissons_ratio=0.35,
        thermal_expansion=20e-6,
    ),
    "Kraft phenolic": MaterialProperties(
        name="Kraft Phenolic",
        density=950,
        youngs_modulus=8e9,
        shear_modulus=3e9,
        yield_strength=60e6,
        ultimate_strength=90e6,
        poissons_ratio=0.35,
        thermal_expansion=25e-6,
    ),
}


@dataclass
class FlutterResult:
    """Fin flutter analysis results"""

    flutter_velocity: float  # m/s - velocity where flutter occurs
    flutter_frequency: float  # Hz - flutter frequency
    safety_margin: float  # ratio of flutter velocity to max velocity
    is_safe: bool  # True if margin > 1.5
    critical_altitude: float  # Altitude where flutter is most likely
    divergence_velocity: float  # Static divergence velocity


@dataclass
class LoadsResult:
    """Structural loads at a given flight condition"""

    time: float
    altitude: float
    velocity: float
    mach: float
    dynamic_pressure: float
    # Loads
    axial_load: float  # N - along rocket axis (thrust - drag)
    shear_load: float  # N - perpendicular to axis
    bending_moment: float  # N·m - about CG
    torsion_moment: float  # N·m - about longitudinal axis
    # Accelerations
    axial_acceleration: float  # g
    lateral_acceleration: float  # g


@dataclass
class StressResult:
    """Stress analysis results for a component"""

    component_name: str
    # Stresses (Pa)
    axial_stress: float
    bending_stress: float
    shear_stress: float
    von_mises_stress: float
    # Margins
    yield_margin: float  # yield_strength / von_mises - 1
    ultimate_margin: float  # ultimate_strength / von_mises - 1
    is_safe: bool


@dataclass
class StructuralAnalysisResult:
    """Complete structural analysis results"""

    flutter: FlutterResult
    max_loads: LoadsResult
    component_stresses: List[StressResult]
    overall_safety_margin: float
    critical_component: str
    recommendations: List[str]


class FinFlutterAnalyzer:
    """
    Fin flutter analysis using NACA TN-4197 method.

    Flutter occurs when aerodynamic forces couple with structural
    vibration modes to create self-sustaining oscillations.

    Critical parameters:
    - Fin aspect ratio
    - Taper ratio
    - Thickness ratio
    - Material stiffness
    - Air density (altitude dependent)
    """

    def __init__(
        self,
        root_chord: float,
        tip_chord: float,
        span: float,
        thickness: float,
        shear_modulus: float,
        body_radius: float = 0.05,
    ):
        self.root_chord = root_chord
        self.tip_chord = tip_chord
        self.span = span
        self.thickness = thickness
        self.shear_modulus = shear_modulus
        self.body_radius = body_radius

    @property
    def aspect_ratio(self) -> float:
        """Geometric aspect ratio AR = 2*span / (root + tip)"""
        return 2 * self.span / (self.root_chord + self.tip_chord)

    @property
    def taper_ratio(self) -> float:
        """Taper ratio λ = tip_chord / root_chord"""
        if self.root_chord > 0:
            return self.tip_chord / self.root_chord
        return 0

    @property
    def thickness_ratio(self) -> float:
        """Thickness ratio t/c at root"""
        if self.root_chord > 0:
            return self.thickness / self.root_chord
        return 0.1

    def calculate_flutter_velocity(
        self,
        altitude: float = 0.0,
        temperature_offset: float = 0.0,
    ) -> FlutterResult:
        """
        Calculate fin flutter velocity using NACA TN-4197.

        V_flutter = a * sqrt(G / (ρ * AR³ * P))

        Where P is a geometric parameter based on taper and thickness.

        Args:
            altitude: Flight altitude in meters
            temperature_offset: Temperature deviation from ISA in Kelvin

        Returns:
            FlutterResult with velocities and safety assessment
        """
        # Atmospheric properties at altitude (ISA model)
        T0, P0, rho0 = 288.15, 101325, 1.225  # Sea level
        L = 0.0065  # Lapse rate K/m

        if altitude < 11000:
            T = T0 - L * altitude + temperature_offset
            P = P0 * (T / T0) ** 5.256
            rho = rho0 * (T / T0) ** 4.256
        else:
            # Simplified isothermal above 11km
            T = 216.65 + temperature_offset
            rho = 0.364 * math.exp(-(altitude - 11000) / 6341)

        # Speed of sound
        gamma = 1.4
        R = 287.05
        a = math.sqrt(gamma * R * T)

        # NACA TN-4197 flutter parameter
        AR = self.aspect_ratio
        t_c = self.thickness_ratio
        G = self.shear_modulus

        # Geometric flutter parameter
        # P = (AR³ * (λ+1) * t_c) / (2 * (AR + 2) * (λ + 1))
        # Simplified form:
        P = (AR**3) * t_c / (2 * (AR + 2))

        # Flutter velocity
        if P > 0 and rho > 0:
            v_flutter = a * math.sqrt(G / (rho * P * 1e6))
            v_flutter = min(v_flutter, 3000)  # Cap at reasonable value
        else:
            v_flutter = 1000  # Default safe value

        # Flutter frequency estimate (torsional mode)
        # f ≈ (1/2π) * sqrt(G*J / (ρ*A*L⁴))
        mean_chord = (self.root_chord + self.tip_chord) / 2
        J = mean_chord * self.thickness**3 / 3  # Torsional constant
        A = mean_chord * self.span  # Planform area

        if A > 0:
            f_flutter = (1 / (2 * math.pi)) * math.sqrt(G * J / (1850 * A * self.span**4))
            f_flutter = max(f_flutter, 10)  # Minimum 10 Hz
        else:
            f_flutter = 50  # Default

        # Divergence velocity (static instability)
        # V_div ≈ 1.4 * V_flutter for typical fins
        v_divergence = 1.4 * v_flutter

        # Find critical altitude (lowest flutter velocity)
        critical_alt = 0
        min_flutter_v = v_flutter
        for alt_test in range(0, 30000, 1000):
            result = self._flutter_at_altitude(alt_test, G, AR, t_c)
            if result < min_flutter_v:
                min_flutter_v = result
                critical_alt = alt_test

        return FlutterResult(
            flutter_velocity=v_flutter,
            flutter_frequency=f_flutter,
            safety_margin=0.0,  # Set by caller with max velocity
            is_safe=True,  # Set by caller
            critical_altitude=critical_alt,
            divergence_velocity=v_divergence,
        )

    def _flutter_at_altitude(self, altitude: float, G: float, AR: float, t_c: float) -> float:
        """Helper to calculate flutter velocity at specific altitude"""
        T0, rho0 = 288.15, 1.225
        L = 0.0065

        if altitude < 11000:
            T = T0 - L * altitude
            rho = rho0 * (T / T0) ** 4.256
        else:
            rho = 0.364 * math.exp(-(altitude - 11000) / 6341)

        gamma, R = 1.4, 287.05
        T = max(T0 - L * min(altitude, 11000), 216.65)
        a = math.sqrt(gamma * R * T)

        P = (AR**3) * t_c / (2 * (AR + 2))

        if P > 0 and rho > 0:
            return a * math.sqrt(G / (rho * P * 1e6))
        return 1000


class StructuralLoadsAnalyzer:
    """
    Calculate structural loads throughout flight.

    Load sources:
    - Thrust (compression during boost)
    - Drag (tension during coast/descent)
    - Inertia (acceleration loads)
    - Aerodynamic moments (bending)
    - Recovery deployment (shock loads)
    """

    def __init__(self, rocket_mass: float, rocket_length: float, reference_area: float):
        self.rocket_mass = rocket_mass
        self.rocket_length = rocket_length
        self.reference_area = reference_area

    def calculate_loads(
        self,
        time: float,
        altitude: float,
        velocity: float,
        acceleration: float,
        thrust: float,
        drag: float,
        angle_of_attack: float,
        angular_velocity: float,
    ) -> LoadsResult:
        """Calculate structural loads at a flight condition"""

        # Atmospheric properties
        if altitude < 11000:
            rho = 1.225 * (1 - 0.0065 * altitude / 288.15) ** 4.256
        else:
            rho = 0.364 * math.exp(-(altitude - 11000) / 6341)

        # Dynamic pressure
        q = 0.5 * rho * velocity**2

        # Mach number
        T = max(288.15 - 0.0065 * min(altitude, 11000), 216.65)
        a = math.sqrt(1.4 * 287.05 * T)
        mach = velocity / a if a > 0 else 0

        # Axial load: thrust - drag - gravity component
        # During boost: compression (positive)
        # During coast: tension from drag (negative)
        axial_load = thrust - drag

        # Shear load from angle of attack
        # Lift ≈ CN_α * α * q * S
        cn_alpha = 2 * math.pi  # Approximate
        lift = cn_alpha * angle_of_attack * q * self.reference_area
        shear_load = abs(lift)

        # Bending moment from lift acting at CP
        # Assume CP at 60% of length from nose
        cp_position = 0.6 * self.rocket_length
        cg_position = 0.5 * self.rocket_length  # Approximate
        moment_arm = abs(cp_position - cg_position)
        bending_moment = shear_load * moment_arm

        # Torsion from fin cant or asymmetric loading
        # Usually small for symmetric rockets
        torsion_moment = 0.0

        # Accelerations
        axial_accel = acceleration / 9.81 if self.rocket_mass > 0 else 0
        lateral_accel = shear_load / (self.rocket_mass * 9.81) if self.rocket_mass > 0 else 0

        return LoadsResult(
            time=time,
            altitude=altitude,
            velocity=velocity,
            mach=mach,
            dynamic_pressure=q,
            axial_load=axial_load,
            shear_load=shear_load,
            bending_moment=bending_moment,
            torsion_moment=torsion_moment,
            axial_acceleration=axial_accel,
            lateral_acceleration=lateral_accel,
        )


class StressAnalyzer:
    """
    Calculate stresses in rocket components.

    Uses classical beam theory and thin-wall pressure vessel equations.
    """

    @staticmethod
    def body_tube_stress(
        outer_radius: float,
        wall_thickness: float,
        axial_load: float,
        bending_moment: float,
        internal_pressure: float,
        material: MaterialProperties,
    ) -> StressResult:
        """
        Calculate stresses in a body tube (thin-walled cylinder).

        Stress sources:
        - Axial load (uniform stress)
        - Bending (max at outer fiber)
        - Internal pressure (hoop and longitudinal)
        """
        inner_radius = outer_radius - wall_thickness

        # Cross-sectional area
        area = math.pi * (outer_radius**2 - inner_radius**2)

        # Second moment of area
        moment_of_inertia = math.pi * (outer_radius**4 - inner_radius**4) / 4

        # Axial stress from load
        axial_stress = axial_load / area if area > 0 else 0

        # Bending stress (max at outer fiber)
        bending_stress = (
            bending_moment * outer_radius / moment_of_inertia if moment_of_inertia > 0 else 0
        )

        # Hoop stress from internal pressure (thin wall: σ = pr/t)
        hoop_stress = internal_pressure * inner_radius / wall_thickness if wall_thickness > 0 else 0

        # Shear stress (assume small for body tube)
        shear_stress = 0

        # Combined stress - von Mises
        # σ_vm = sqrt(σ_x² - σ_x*σ_y + σ_y² + 3τ²)
        sigma_x = axial_stress + bending_stress
        sigma_y = hoop_stress
        von_mises = math.sqrt(sigma_x**2 - sigma_x * sigma_y + sigma_y**2 + 3 * shear_stress**2)

        # Safety margins
        yield_margin = material.yield_strength / von_mises - 1 if von_mises > 0 else 100
        ultimate_margin = material.ultimate_strength / von_mises - 1 if von_mises > 0 else 100

        return StressResult(
            component_name="Body Tube",
            axial_stress=axial_stress,
            bending_stress=bending_stress,
            shear_stress=shear_stress,
            von_mises_stress=von_mises,
            yield_margin=yield_margin,
            ultimate_margin=ultimate_margin,
            is_safe=yield_margin > 0.5,  # 50% margin required
        )

    @staticmethod
    def fin_stress(
        root_chord: float,
        span: float,
        thickness: float,
        shear_load: float,
        material: MaterialProperties,
    ) -> StressResult:
        """
        Calculate stresses in a fin (cantilever beam model).

        Max stress at root due to bending from aerodynamic loads.
        """
        # Section modulus for rectangular cross-section
        # S = b*h²/6 where b=chord, h=thickness
        section_modulus = root_chord * thickness**2 / 6

        # Bending moment at root (load at span/2 from root)
        moment_arm = span / 2
        bending_moment = shear_load * moment_arm

        # Bending stress
        bending_stress = bending_moment / section_modulus if section_modulus > 0 else 0

        # Shear stress (average)
        area = root_chord * thickness
        shear_stress = 1.5 * shear_load / area if area > 0 else 0  # 1.5 for rectangular section

        # Von Mises
        von_mises = math.sqrt(bending_stress**2 + 3 * shear_stress**2)

        # Margins
        yield_margin = material.yield_strength / von_mises - 1 if von_mises > 0 else 100
        ultimate_margin = material.ultimate_strength / von_mises - 1 if von_mises > 0 else 100

        return StressResult(
            component_name="Fins",
            axial_stress=0,
            bending_stress=bending_stress,
            shear_stress=shear_stress,
            von_mises_stress=von_mises,
            yield_margin=yield_margin,
            ultimate_margin=ultimate_margin,
            is_safe=yield_margin > 0.5,
        )


class StructuralAnalyzer:
    """
    Complete structural analysis for a rocket.

    Integrates flutter, loads, and stress analysis.
    """

    def __init__(self, rocket, motor):
        """
        Args:
            rocket: Rocket object with geometry
            motor: Motor object with thrust curve
        """
        self.rocket = rocket
        self.motor = motor

    def analyze(self, flight_history: List) -> StructuralAnalysisResult:
        """
        Perform complete structural analysis.

        Args:
            flight_history: List of StateSnapshot from flight simulation

        Returns:
            StructuralAnalysisResult with all analysis data
        """
        recommendations = []

        # Extract rocket geometry
        body_radius = self.rocket.reference_diameter / 2
        body_length = self.rocket.reference_length

        # Find fins for flutter analysis
        fin_data = self._find_fin_data()

        # Get material properties
        body_material = STRUCTURAL_MATERIALS.get("Fiberglass", STRUCTURAL_MATERIALS["Fiberglass"])
        fin_material = STRUCTURAL_MATERIALS.get("Fiberglass", STRUCTURAL_MATERIALS["Fiberglass"])

        # Flutter analysis
        if fin_data:
            flutter_analyzer = FinFlutterAnalyzer(
                root_chord=fin_data["root_chord"],
                tip_chord=fin_data["tip_chord"],
                span=fin_data["span"],
                thickness=fin_data["thickness"],
                shear_modulus=fin_material.shear_modulus,
                body_radius=body_radius,
            )
            flutter_result = flutter_analyzer.calculate_flutter_velocity()

            # Check against max velocity
            max_velocity = max(s.speed for s in flight_history) if flight_history else 0
            flutter_result.safety_margin = (
                flutter_result.flutter_velocity / max_velocity if max_velocity > 0 else 10
            )
            flutter_result.is_safe = flutter_result.safety_margin > 1.5

            if not flutter_result.is_safe:
                recommendations.append(
                    f"⚠️ Fin flutter risk: Max velocity {max_velocity:.0f} m/s exceeds "
                    f"safe limit ({flutter_result.flutter_velocity / 1.5:.0f} m/s). "
                    "Consider thicker fins or stiffer material."
                )
        else:
            flutter_result = FlutterResult(
                flutter_velocity=1000,
                flutter_frequency=50,
                safety_margin=10,
                is_safe=True,
                critical_altitude=0,
                divergence_velocity=1400,
            )

        # Find max loads condition
        reference_area = math.pi * body_radius**2
        loads_analyzer = StructuralLoadsAnalyzer(
            rocket_mass=self._get_dry_mass(),
            rocket_length=body_length,
            reference_area=reference_area,
        )

        max_loads = None
        max_q = 0

        for snapshot in flight_history:
            if snapshot.dynamic_pressure > max_q:
                max_q = snapshot.dynamic_pressure
                velocity = snapshot.speed
                acceleration = (
                    np.linalg.norm(snapshot.acceleration)
                    if hasattr(snapshot, "acceleration")
                    else 50
                )
                thrust = (
                    self.motor.thrust(snapshot.time) if snapshot.time < self.motor.burn_time else 0
                )
                drag = np.linalg.norm(snapshot.drag_force)

                max_loads = loads_analyzer.calculate_loads(
                    time=snapshot.time,
                    altitude=snapshot.position[2],
                    velocity=velocity,
                    acceleration=acceleration,
                    thrust=thrust,
                    drag=drag,
                    angle_of_attack=snapshot.angle_of_attack,
                    angular_velocity=np.linalg.norm(snapshot.angular_velocity),
                )

        if max_loads is None:
            max_loads = LoadsResult(
                time=0,
                altitude=0,
                velocity=0,
                mach=0,
                dynamic_pressure=0,
                axial_load=0,
                shear_load=0,
                bending_moment=0,
                torsion_moment=0,
                axial_acceleration=0,
                lateral_acceleration=0,
            )

        # Stress analysis
        component_stresses = []

        # Body tube stress
        wall_thickness = 0.002  # Assume 2mm wall
        body_stress = StressAnalyzer.body_tube_stress(
            outer_radius=body_radius,
            wall_thickness=wall_thickness,
            axial_load=max_loads.axial_load,
            bending_moment=max_loads.bending_moment,
            internal_pressure=0,  # Assume vented
            material=body_material,
        )
        component_stresses.append(body_stress)

        if not body_stress.is_safe:
            recommendations.append(
                f"⚠️ Body tube stress margin low ({body_stress.yield_margin:.1%}). "
                "Consider thicker wall or stronger material."
            )

        # Fin stress
        if fin_data:
            # Estimate fin load from max dynamic pressure
            fin_area = 0.5 * (fin_data["root_chord"] + fin_data["tip_chord"]) * fin_data["span"]
            fin_load = max_q * fin_area * 0.1  # Approximate load coefficient

            fin_stress = StressAnalyzer.fin_stress(
                root_chord=fin_data["root_chord"],
                span=fin_data["span"],
                thickness=fin_data["thickness"],
                shear_load=fin_load,
                material=fin_material,
            )
            component_stresses.append(fin_stress)

            if not fin_stress.is_safe:
                recommendations.append(
                    f"⚠️ Fin stress margin low ({fin_stress.yield_margin:.1%}). "
                    "Consider thicker fins or adding fillets."
                )

        # Overall safety
        min_margin = min(s.yield_margin for s in component_stresses) if component_stresses else 10
        critical_component = (
            min(component_stresses, key=lambda s: s.yield_margin).component_name
            if component_stresses
            else "None"
        )

        if min_margin > 2.0:
            recommendations.append("✓ All structural margins adequate (>200%)")
        elif min_margin > 1.0:
            recommendations.append("⚠️ Structural margins marginal (100-200%)")
        else:
            recommendations.append("❌ STRUCTURAL FAILURE RISK - margins below 100%")

        return StructuralAnalysisResult(
            flutter=flutter_result,
            max_loads=max_loads,
            component_stresses=component_stresses,
            overall_safety_margin=min_margin,
            critical_component=critical_component,
            recommendations=recommendations,
        )

    def _find_fin_data(self) -> Optional[Dict]:
        """Extract fin geometry from rocket"""

        def traverse(component):
            # Check for fin sets
            if hasattr(component, "root_chord") and hasattr(component, "span"):
                return {
                    "root_chord": component.root_chord,
                    "tip_chord": getattr(component, "tip_chord", component.root_chord * 0.5),
                    "span": component.span,
                    "thickness": getattr(component, "thickness", 0.003),
                }

            for child in getattr(component, "children", []):
                result = traverse(child)
                if result:
                    return result
            return None

        return traverse(self.rocket)

    def _get_dry_mass(self) -> float:
        """Get rocket dry mass"""

        def sum_mass(component):
            mass = component.get_mass() if hasattr(component, "get_mass") else 0
            for child in getattr(component, "children", []):
                mass += sum_mass(child)
            return mass

        return sum_mass(self.rocket)


# Alias for backwards compatibility and clearer naming
MATERIAL_PROPERTIES = STRUCTURAL_MATERIALS
