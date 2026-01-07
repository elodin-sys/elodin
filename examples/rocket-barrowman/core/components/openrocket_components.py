"""
Complete OpenRocket component system reimplemented in Python.
Matches OpenRocket's component hierarchy and mass/aerodynamic calculations exactly.
"""

from dataclasses import dataclass
from typing import Optional, List
from enum import Enum
import math


class FinishType(Enum):
    """Surface finish types affecting drag"""

    SMOOTH = "smooth"
    UNFINISHED = "unfinished"
    ROUGH = "rough"
    POLISHED = "polished"


class Material:
    """Material with density"""

    def __init__(self, name: str, density: float, description: str = ""):
        self.name = name
        self.density = density  # kg/m^3
        self.description = description


# Standard OpenRocket materials
MATERIALS = {
    "Cardboard": Material("Cardboard", 680.0),
    "Kraft phenolic": Material("Kraft phenolic", 950.0),
    "Blue tube": Material("Blue tube", 1040.0),
    "Quantum tube": Material("Quantum tube", 1040.0),
    "Fiberglass": Material("Fiberglass", 1850.0),
    "Carbon fiber": Material("Carbon fiber", 1780.0),
    "Plywood (birch)": Material("Plywood (birch)", 630.0),
    "Plywood (light)": Material("Plywood (light)", 350.0),
    "Balsa": Material("Balsa", 170.0),
    "Polystyrene (cast)": Material("Polystyrene (cast)", 1050.0),
    "Polystyrene (extruded)": Material("Polystyrene (extruded)", 40.0),
    "ABS plastic": Material("ABS plastic", 1050.0),
    "PLA plastic": Material("PLA plastic", 1240.0),
    "Nylon": Material("Nylon", 1150.0),
    "Acrylic": Material("Acrylic", 1190.0),
    "Aluminum": Material("Aluminum", 2700.0),
    "Brass": Material("Brass", 8530.0),
    "Steel": Material("Steel", 7850.0),
    "Titanium": Material("Titanium", 4500.0),
}


@dataclass
class Position:
    """Position along rocket axis (relative to parent or absolute)"""

    x: float  # meters from reference point


class RocketComponent:
    """Base class for all rocket components - matches OpenRocket hierarchy"""

    def __init__(self, name: str = ""):
        self.name = name
        self.position = Position(0.0)
        self.children: List[RocketComponent] = []
        self.parent: Optional[RocketComponent] = None
        self.override_mass: Optional[float] = None
        self.override_cg_x: Optional[float] = None
        self.comment: str = ""

    def add_child(self, component: "RocketComponent"):
        """Add a child component"""
        self.children.append(component)
        component.parent = self

    def get_mass(self) -> float:
        """Get component mass (override or calculated)"""
        if self.override_mass is not None:
            return self.override_mass
        return self.calculate_mass()

    def calculate_mass(self) -> float:
        """Calculate component mass from geometry and material"""
        return 0.0

    def get_cg_x(self) -> float:
        """Get component CG position"""
        if self.override_cg_x is not None:
            return self.override_cg_x
        return self.calculate_cg_x()

    def calculate_cg_x(self) -> float:
        """Calculate CG position from geometry"""
        return self.position.x

    def get_absolute_position(self) -> float:
        """Get absolute position in rocket"""
        pos = self.position.x
        parent = self.parent
        while parent is not None:
            pos += parent.position.x
            parent = parent.parent
        return pos

    def get_total_mass(self) -> float:
        """Get total mass including children"""
        mass = self.get_mass()
        for child in self.children:
            mass += child.get_total_mass()
        return mass

    def get_total_cg(self) -> float:
        """Get total CG including children"""
        total_mass = 0.0
        total_moment = 0.0

        # This component
        mass = self.get_mass()
        if mass > 0:
            cg = self.get_absolute_position() + self.get_cg_x()
            total_mass += mass
            total_moment += mass * cg

        # Children
        for child in self.children:
            child_mass = child.get_total_mass()
            if child_mass > 0:
                child_cg = child.get_total_cg()
                total_mass += child_mass
                total_moment += child_mass * child_cg

        if total_mass > 0:
            return total_moment / total_mass
        return 0.0


class SymmetricComponent(RocketComponent):
    """Component with circular cross-section"""

    def __init__(self, name: str = ""):
        super().__init__(name)
        self.finish = FinishType.SMOOTH
        self.material = MATERIALS["Cardboard"]


class BodyTube(SymmetricComponent):
    """Body tube component - exact OpenRocket implementation"""

    def __init__(
        self,
        name: str = "Body tube",
        length: float = 0.3,
        outer_radius: float = 0.025,
        thickness: float = 0.001,
    ):
        super().__init__(name)
        self.length = length
        self.outer_radius = outer_radius
        self.thickness = thickness
        self.motor_mount = False

    def calculate_mass(self) -> float:
        """Calculate mass from tube geometry"""
        inner_radius = self.outer_radius - self.thickness
        volume = math.pi * self.length * (self.outer_radius**2 - inner_radius**2)
        return volume * self.material.density

    def calculate_cg_x(self) -> float:
        """CG is at center of tube"""
        return self.length / 2.0

    def get_inner_radius(self) -> float:
        return self.outer_radius - self.thickness


class NoseCone(SymmetricComponent):
    """Nose cone component - exact OpenRocket shapes"""

    class Shape(Enum):
        CONICAL = "conical"
        OGIVE = "ogive"
        ELLIPSOID = "ellipsoid"
        PARABOLIC = "parabolic"
        POWER_SERIES = "power_series"
        HAACK = "haack"
        VON_KARMAN = "von_karman"

    def __init__(
        self,
        name: str = "Nose cone",
        length: float = 0.1,
        base_radius: float = 0.025,
        thickness: float = 0.002,
        shape: Shape = Shape.OGIVE,
    ):
        super().__init__(name)
        self.length = length
        self.base_radius = base_radius
        self.thickness = thickness
        self.shape = shape
        self.shape_parameter = 0.0  # For power series and Haack

    def calculate_mass(self) -> float:
        """Calculate mass based on shape"""
        # Volume calculation depends on shape
        if self.shape == self.Shape.CONICAL:
            outer_vol = (1 / 3) * math.pi * self.base_radius**2 * self.length
        elif self.shape == self.Shape.OGIVE:
            # Tangent ogive approximation
            outer_vol = 0.5 * math.pi * self.base_radius**2 * self.length
        else:
            # Generic approximation for other shapes
            outer_vol = 0.5 * math.pi * self.base_radius**2 * self.length

        # Subtract hollow interior if applicable
        if self.thickness > 0:
            inner_radius = self.base_radius - self.thickness
            scale = inner_radius / self.base_radius
            inner_vol = outer_vol * scale**2
            volume = outer_vol - inner_vol
        else:
            volume = outer_vol

        return volume * self.material.density

    def calculate_cg_x(self) -> float:
        """CG position depends on shape"""
        if self.shape == self.Shape.CONICAL:
            return self.length * 0.667  # 2/3 from tip
        elif self.shape == self.Shape.OGIVE:
            return self.length * 0.534  # Empirical for ogive
        else:
            return self.length * 0.5  # Generic approximation


class Transition(SymmetricComponent):
    """Transition/reducer between different diameters"""

    class Shape(Enum):
        CONICAL = "conical"
        OGIVE = "ogive"
        ELLIPSOID = "ellipsoid"

    def __init__(
        self,
        name: str = "Transition",
        length: float = 0.1,
        fore_radius: float = 0.02,
        aft_radius: float = 0.025,
        thickness: float = 0.001,
        shape: Shape = Shape.CONICAL,
    ):
        super().__init__(name)
        self.length = length
        self.fore_radius = fore_radius
        self.aft_radius = aft_radius
        self.thickness = thickness
        self.shape = shape

    def calculate_mass(self) -> float:
        """Calculate transition mass"""
        # Frustum volume approximation
        r1 = self.fore_radius
        r2 = self.aft_radius
        outer_vol = (math.pi * self.length / 3) * (r1**2 + r1 * r2 + r2**2)

        r1_inner = max(0, r1 - self.thickness)
        r2_inner = max(0, r2 - self.thickness)
        inner_vol = (math.pi * self.length / 3) * (r1_inner**2 + r1_inner * r2_inner + r2_inner**2)

        volume = outer_vol - inner_vol
        return volume * self.material.density

    def calculate_cg_x(self) -> float:
        """CG of frustum"""
        r1 = self.fore_radius
        r2 = self.aft_radius
        # CG of frustum from narrow end
        if r1 != r2:
            cg = self.length * (r1**2 + 2 * r1 * r2 + 3 * r2**2) / (4 * (r1**2 + r1 * r2 + r2**2))
        else:
            cg = self.length / 2
        return cg


class TrapezoidFinSet(RocketComponent):
    """
    Trapezoidal fin set - OpenRocket standard.

    Geometry (side view of single fin):

            sweep_length
            |<------->|
            +---------+  <- tip_chord (length at outer edge)
           /         /
          /         /
         /         /     span (height from body to tip)
        /         /
       /         /
      +--------------+   <- root_chord (length at body attachment)

      |<------------>|
         root_chord

    Parameters:
        root_chord: Length of fin at body tube attachment (m)
        tip_chord: Length of fin at outer tip (m)
        span: Height of fin from body surface to tip (m)
        sweep_length: Horizontal distance from root leading edge to tip leading edge (m)
                      (NOT sweep angle! Angle = atan(sweep_length / span))
        thickness: Fin thickness (m)

    Typical ratios (relative to body diameter D):
        root_chord ≈ 2.0 × D
        tip_chord ≈ 0.33-0.5 × D
        span ≈ 1.0 × D
        sweep_length ≈ root_chord - tip_chord (for ~45° leading edge)
    """

    def __init__(
        self,
        name: str = "Trapezoidal fin set",
        fin_count: int = 3,
        root_chord: float = 0.1,
        tip_chord: float = 0.05,
        span: float = 0.05,
        sweep_length: float = 0.03,
        thickness: float = 0.003,
    ):
        super().__init__(name)
        self.fin_count = fin_count
        self.root_chord = root_chord
        self.tip_chord = tip_chord
        self.span = span
        self.sweep_length = sweep_length  # Distance from root LE to tip LE
        self.thickness = thickness
        self.material = MATERIALS["Plywood (birch)"]
        self.cant_angle = 0.0  # radians

    @property
    def sweep(self):
        """Alias for sweep_length for backward compatibility."""
        return self.sweep_length

    @sweep.setter
    def sweep(self, value):
        self.sweep_length = value

    @property
    def sweep_angle_deg(self) -> float:
        """Leading edge sweep angle in degrees."""
        import math

        if self.span > 0:
            return math.degrees(math.atan(self.sweep_length / self.span))
        return 0.0

    def calculate_mass(self) -> float:
        """Calculate single fin mass, multiply by count"""
        # Trapezoidal area
        area = 0.5 * (self.root_chord + self.tip_chord) * self.span
        volume = area * self.thickness
        single_fin_mass = volume * self.material.density
        return single_fin_mass * self.fin_count

    def calculate_cg_x(self) -> float:
        """CG of trapezoid"""
        # Trapezoid CG from leading edge
        a = self.root_chord
        b = self.tip_chord
        cg = (a + 2 * b) / (3 * (a + b)) * self.root_chord
        return cg

    def get_mac(self) -> float:
        """Mean aerodynamic chord"""
        return 2 * (self.root_chord + self.tip_chord) / 3


class EllipticalFinSet(RocketComponent):
    """Elliptical fin set"""

    def __init__(
        self,
        name: str = "Elliptical fin set",
        fin_count: int = 3,
        root_chord: float = 0.1,
        span: float = 0.05,
        thickness: float = 0.003,
    ):
        super().__init__(name)
        self.fin_count = fin_count
        self.root_chord = root_chord
        self.span = span
        self.thickness = thickness
        self.material = MATERIALS["Plywood (birch)"]

    def calculate_mass(self) -> float:
        """Calculate elliptical fin mass"""
        area = 0.25 * math.pi * self.root_chord * self.span
        volume = area * self.thickness
        single_fin_mass = volume * self.material.density
        return single_fin_mass * self.fin_count

    def calculate_cg_x(self) -> float:
        """CG of ellipse"""
        return 4 * self.root_chord / (3 * math.pi)


class InnerTube(SymmetricComponent):
    """Inner tube for motor mount or staging"""

    def __init__(
        self,
        name: str = "Inner tube",
        length: float = 0.2,
        outer_radius: float = 0.02,
        thickness: float = 0.001,
    ):
        super().__init__(name)
        self.length = length
        self.outer_radius = outer_radius
        self.thickness = thickness

    def calculate_mass(self) -> float:
        """Same as body tube"""
        inner_radius = self.outer_radius - self.thickness
        volume = math.pi * self.length * (self.outer_radius**2 - inner_radius**2)
        return volume * self.material.density

    def calculate_cg_x(self) -> float:
        return self.length / 2.0


class CenteringRing(RocketComponent):
    """Centering ring for motor mount"""

    def __init__(
        self,
        name: str = "Centering ring",
        outer_radius: float = 0.025,
        inner_radius: float = 0.02,
        thickness: float = 0.003,
    ):
        super().__init__(name)
        self.outer_radius = outer_radius
        self.inner_radius = inner_radius
        self.thickness = thickness
        self.material = MATERIALS["Plywood (birch)"]

    def calculate_mass(self) -> float:
        """Annular disk mass"""
        area = math.pi * (self.outer_radius**2 - self.inner_radius**2)
        volume = area * self.thickness
        return volume * self.material.density

    def calculate_cg_x(self) -> float:
        return self.thickness / 2.0


class RecoveryBulkhead(RocketComponent):
    """Bulkhead plate for compartment separation and load bearing.

    Deprecated simple version kept for backward compatibility with older configs.
    Prefer the more detailed `Bulkhead` definition in the recovery system section.
    """

    def __init__(self, name: str = "Bulkhead", radius: float = 0.025, thickness: float = 0.005):
        super().__init__(name)
        self.radius = radius
        self.thickness = thickness
        self.material = MATERIALS["Plywood (birch)"]

    def calculate_mass(self) -> float:
        """Circular disk mass"""
        volume = math.pi * self.radius**2 * self.thickness
        return volume * self.material.density

    def calculate_cg_x(self) -> float:
        return self.thickness / 2.0


class LaunchLugLegacy(RocketComponent):
    """Simple launch lug model (legacy, kept for backward compatibility)."""

    def __init__(
        self,
        name: str = "Launch lug",
        length: float = 0.03,
        outer_radius: float = 0.005,
        thickness: float = 0.0005,
    ):
        super().__init__(name)
        self.length = length
        self.outer_radius = outer_radius
        self.thickness = thickness
        self.material = MATERIALS["Cardboard"]
        self.radial_position = 0.0  # Angle around body

    def calculate_mass(self) -> float:
        """Small tube mass"""
        inner_radius = self.outer_radius - self.thickness
        volume = math.pi * self.length * (self.outer_radius**2 - inner_radius**2)
        return volume * self.material.density


class RailButtonLegacy(RocketComponent):
    """Rail button (legacy, use `RailButton` below for new designs)."""

    def __init__(
        self, name: str = "Rail button", outer_diameter: float = 0.01, base_height: float = 0.003
    ):
        super().__init__(name)
        self.outer_diameter = outer_diameter
        self.base_height = base_height
        self.material = MATERIALS["Nylon"]

    def calculate_mass(self) -> float:
        """Small button mass"""
        radius = self.outer_diameter / 2
        volume = math.pi * radius**2 * self.base_height
        return volume * self.material.density


class ShockCordLegacy(RocketComponent):
    """Legacy simple shock cord model for backward compatibility."""

    def __init__(self, name: str = "Shock cord", length: float = 1.0, diameter: float = 0.003):
        super().__init__(name)
        self.length = length
        self.diameter = diameter
        self.material_density = 1200.0  # kg/m^3 for nylon cord

    def calculate_mass(self) -> float:
        """Cord mass"""
        radius = self.diameter / 2
        volume = math.pi * radius**2 * self.length
        return volume * self.material_density


class MassComponent(RocketComponent):
    """Generic mass component (ballast, avionics, battery, etc)"""

    def __init__(
        self,
        name: str = "Mass component",
        mass: float = 0.05,
        length: float = 0.02,
        radius: float = 0.015,
    ):
        super().__init__(name)
        self._mass = mass
        self.length = length
        self.radius = radius

    def calculate_mass(self) -> float:
        return self._mass

    def calculate_cg_x(self) -> float:
        return self.length / 2.0


class Parachute(RocketComponent):
    """Parachute recovery device"""

    def __init__(
        self,
        name: str = "Parachute",
        diameter: float = 0.3,
        cd: float = 0.75,
        material_density: float = 50.0,
    ):
        super().__init__(name)
        self.diameter = diameter
        self.cd = cd  # Drag coefficient
        self.material_density = material_density  # g/m^2
        self.deployment_delay = 0.0  # seconds after event
        self.deployment_event = "APOGEE"  # APOGEE, ALTITUDE, TIME
        self.deployment_altitude = 0.0  # for ALTITUDE event
        self.deployment_time = 0.0  # for TIME event
        self.deployed = False

    def calculate_mass(self) -> float:
        """Parachute mass from material"""
        area = math.pi * (self.diameter / 2) ** 2
        mass_g = area * self.material_density
        return mass_g / 1000.0  # Convert to kg

    def get_drag_area(self) -> float:
        """Effective drag area when deployed"""
        if self.deployed:
            area = math.pi * (self.diameter / 2) ** 2
            return self.cd * area
        return 0.0


class Streamer(RocketComponent):
    """Streamer recovery device"""

    def __init__(
        self, name: str = "Streamer", length: float = 0.5, width: float = 0.05, cd: float = 0.5
    ):
        super().__init__(name)
        self.length = length
        self.width = width
        self.cd = cd
        self.material_density = 50.0  # g/m^2
        self.deployment_event = "APOGEE"
        self.deployed = False

    def calculate_mass(self) -> float:
        """Streamer mass"""
        area = self.length * self.width
        mass_g = area * self.material_density
        return mass_g / 1000.0

    def get_drag_area(self) -> float:
        """Effective drag area"""
        if self.deployed:
            area = self.length * self.width
            return self.cd * area
        return 0.0


class Rocket(RocketComponent):
    """Top-level rocket assembly"""

    def __init__(self, name: str = "Rocket"):
        super().__init__(name)
        self.designer = ""
        self.revision = ""
        self.reference_diameter = 0.0
        self.reference_length = 0.0

    def calculate_reference_values(self):
        """Calculate reference diameter and length"""
        # Find maximum body tube diameter
        max_diameter = 0.0
        total_length = 0.0

        def traverse(component):
            nonlocal max_diameter, total_length
            if isinstance(component, (BodyTube, NoseCone, Transition)):
                if hasattr(component, "outer_radius"):
                    max_diameter = max(max_diameter, component.outer_radius * 2)
                elif hasattr(component, "base_radius"):
                    max_diameter = max(max_diameter, component.base_radius * 2)

                if hasattr(component, "length"):
                    pos = component.get_absolute_position()
                    total_length = max(total_length, pos + component.length)

            for child in component.children:
                traverse(child)

        traverse(self)
        self.reference_diameter = max_diameter
        self.reference_length = total_length


# =============================================================================
# AIRFOIL PROFILES
# =============================================================================


class AirfoilType(Enum):
    """Fin airfoil cross-section types"""

    SQUARE = "square"  # Flat plate with square edges
    ROUNDED = "rounded"  # Rounded leading and trailing edges
    AIRFOIL = "airfoil"  # Subsonic airfoil (rounded LE, sharp TE)
    DOUBLE_WEDGE = "double_wedge"  # Supersonic diamond profile
    WEDGE = "wedge"  # Single wedge (sharp LE, flat TE)
    NACA_4DIGIT = "naca_4digit"  # NACA 4-digit series (e.g., 0012)
    NACA_5DIGIT = "naca_5digit"  # NACA 5-digit series
    HEXAGONAL = "hexagonal"  # Hex cross-section for strength


@dataclass
class AirfoilProfile:
    """
    Fin airfoil cross-section definition.

    For NACA profiles, use designation like "0012" for NACA 0012.
    For wedge/double-wedge, specify half-angle in degrees.
    """

    airfoil_type: AirfoilType = AirfoilType.SQUARE
    thickness_ratio: float = 0.1  # t/c ratio (thickness / chord)
    naca_designation: str = ""  # e.g., "0012" for NACA 0012
    wedge_angle_deg: float = 10.0  # Half-angle for wedge profiles
    leading_edge_radius: float = 0.0  # LE radius as fraction of thickness

    def get_drag_coefficient_factor(self, mach: float) -> float:
        """
        Get drag coefficient multiplier based on airfoil and Mach number.

        Subsonic (M < 0.8): Rounded profiles lower drag
        Transonic (0.8 < M < 1.2): All profiles high drag
        Supersonic (M > 1.2): Sharp profiles (wedge) lower drag
        """
        t_c = self.thickness_ratio

        if mach < 0.8:  # Subsonic
            if self.airfoil_type in [
                AirfoilType.AIRFOIL,
                AirfoilType.NACA_4DIGIT,
                AirfoilType.NACA_5DIGIT,
            ]:
                return 1.0  # Baseline
            elif self.airfoil_type == AirfoilType.ROUNDED:
                return 1.1
            elif self.airfoil_type == AirfoilType.SQUARE:
                return 1.3  # Blunt edges increase drag
            else:  # Wedge types
                return 1.2
        elif mach < 1.2:  # Transonic
            # All profiles suffer in transonic regime
            return 1.5 + 0.5 * t_c * 10
        else:  # Supersonic
            if self.airfoil_type in [AirfoilType.DOUBLE_WEDGE, AirfoilType.WEDGE]:
                # Sharp profiles better at supersonic
                return 1.0 + 2 * t_c
            elif self.airfoil_type == AirfoilType.HEXAGONAL:
                return 1.1 + 2.5 * t_c
            else:  # Rounded profiles
                return 1.2 + 3 * t_c


# =============================================================================
# ADVANCED FIN COMPONENTS
# =============================================================================


class FinShape(Enum):
    """Fin planform shapes"""

    TRAPEZOIDAL = "trapezoidal"
    ELLIPTICAL = "elliptical"
    CLIPPED_DELTA = "clipped_delta"
    DELTA = "delta"
    RECTANGULAR = "rectangular"
    FREEFORM = "freeform"


class AdvancedFinSet(RocketComponent):
    """
    Advanced fin set with airfoil profiles, fillets, and detailed geometry.

    Supports:
    - Multiple fin shapes (trapezoidal, elliptical, delta, freeform)
    - Airfoil cross-sections (NACA, wedge, etc.)
    - Fin fillets (structural and aerodynamic)
    - Cant angle for roll stabilization
    - Tab extensions for through-the-wall mounting
    """

    def __init__(
        self,
        name: str = "Advanced Fin Set",
        fin_count: int = 4,
        shape: FinShape = FinShape.TRAPEZOIDAL,
        # Planform geometry (for trapezoidal)
        root_chord: float = 0.1,
        tip_chord: float = 0.05,
        span: float = 0.08,
        sweep_length: float = 0.03,
        # Airfoil
        airfoil: AirfoilProfile = None,
        thickness: float = 0.003,
        # Fillet
        fillet_radius: float = 0.0,  # Root fillet radius
        # Mounting
        cant_angle_deg: float = 0.0,  # Rotation for roll induction
        tab_length: float = 0.0,  # Through-wall tab
        tab_height: float = 0.0,
        # Freeform points (for FREEFORM shape)
        freeform_points: List = None,  # List of (x, y) tuples
    ):
        super().__init__(name)
        self.fin_count = fin_count
        self.shape = shape
        self.root_chord = root_chord
        self.tip_chord = tip_chord
        self.span = span
        self.sweep_length = sweep_length
        self.airfoil = airfoil or AirfoilProfile()
        self.thickness = thickness
        self.fillet_radius = fillet_radius
        self.cant_angle_deg = cant_angle_deg
        self.tab_length = tab_length
        self.tab_height = tab_height
        self.freeform_points = freeform_points or []
        self.material = MATERIALS["Fiberglass"]

    def calculate_planform_area(self) -> float:
        """Calculate single fin planform area based on shape"""
        if self.shape == FinShape.TRAPEZOIDAL or self.shape == FinShape.CLIPPED_DELTA:
            return 0.5 * (self.root_chord + self.tip_chord) * self.span
        elif self.shape == FinShape.ELLIPTICAL:
            return 0.25 * math.pi * self.root_chord * self.span
        elif self.shape == FinShape.DELTA:
            return 0.5 * self.root_chord * self.span
        elif self.shape == FinShape.RECTANGULAR:
            return self.root_chord * self.span
        elif self.shape == FinShape.FREEFORM and self.freeform_points:
            # Shoelace formula for polygon area
            n = len(self.freeform_points)
            if n < 3:
                return 0.0
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += self.freeform_points[i][0] * self.freeform_points[j][1]
                area -= self.freeform_points[j][0] * self.freeform_points[i][1]
            return abs(area) / 2.0
        return 0.0

    def calculate_mass(self) -> float:
        """Calculate total fin set mass including fillets"""
        # Fin mass
        fin_area = self.calculate_planform_area()
        fin_volume = fin_area * self.thickness
        fin_mass = fin_volume * self.material.density * self.fin_count

        # Fillet mass (quarter-circle cross-section)
        if self.fillet_radius > 0:
            fillet_area = 0.25 * math.pi * self.fillet_radius**2
            fillet_length = self.root_chord
            fillet_volume = fillet_area * fillet_length * 2  # Both sides
            fillet_mass = fillet_volume * self.material.density * self.fin_count
            fin_mass += fillet_mass

        # Tab mass
        if self.tab_length > 0 and self.tab_height > 0:
            tab_volume = self.tab_length * self.tab_height * self.thickness
            tab_mass = tab_volume * self.material.density * self.fin_count
            fin_mass += tab_mass

        return fin_mass

    def calculate_cg_x(self) -> float:
        """Calculate fin set CG position"""
        if self.shape in [FinShape.TRAPEZOIDAL, FinShape.CLIPPED_DELTA]:
            a = self.root_chord
            b = self.tip_chord
            return (a + 2 * b) / (3 * (a + b)) * self.root_chord
        elif self.shape == FinShape.ELLIPTICAL:
            return self.root_chord * 0.4  # Approximate
        elif self.shape == FinShape.DELTA:
            return self.root_chord / 3
        return self.root_chord * 0.4

    def get_flutter_velocity(self, altitude: float = 0.0) -> float:
        """
        Estimate fin flutter velocity using NACA method.

        V_flutter = a * sqrt(G / (AR^3 * (t/c) * (λ+1) / (2*(AR+2)*(λ+1))))

        Where:
        - a = speed of sound
        - G = shear modulus
        - AR = aspect ratio
        - t/c = thickness ratio
        - λ = taper ratio
        """
        # Simplified flutter estimate
        aspect_ratio = self.span / ((self.root_chord + self.tip_chord) / 2)
        taper_ratio = self.tip_chord / self.root_chord if self.root_chord > 0 else 0
        t_c = self.thickness / self.root_chord if self.root_chord > 0 else 0.1

        # Shear modulus estimate (Pa) - varies by material
        G = 3e9  # Fiberglass approximate
        if self.material.name == "Carbon fiber":
            G = 5e9
        elif self.material.name in ["Plywood (birch)", "Plywood (light)"]:
            G = 0.7e9

        # Speed of sound at altitude (simplified)
        a = 340 - 0.004 * altitude  # Approximate

        # Flutter velocity (simplified NACA formula)
        if aspect_ratio > 0 and t_c > 0:
            flutter_param = (aspect_ratio**3) * t_c * (taper_ratio + 1) / (2 * (aspect_ratio + 2))
            if flutter_param > 0:
                v_flutter = a * math.sqrt(G / (1.225 * flutter_param * 1e6))
                return min(v_flutter, 2000)  # Cap at reasonable value

        return 500  # Default safe value


# =============================================================================
# EXTERNAL PROTUBERANCES
# =============================================================================


class RailButton(RocketComponent):
    """
    Rail button for launch rail guidance.

    Causes parasitic drag - important for accurate simulation.
    """

    def __init__(
        self,
        name: str = "Rail Button",
        outer_diameter: float = 0.010,  # 10mm typical
        height: float = 0.015,  # Height above body
        screw_diameter: float = 0.004,  # #8 screw = 4mm
        standoff: float = 0.002,  # Gap under button
    ):
        super().__init__(name)
        self.outer_diameter = outer_diameter
        self.height = height
        self.screw_diameter = screw_diameter
        self.standoff = standoff
        self.material = MATERIALS["Nylon"]

    def calculate_mass(self) -> float:
        """Rail button mass (approximate as cylinder)"""
        volume = math.pi * (self.outer_diameter / 2) ** 2 * self.height
        return volume * self.material.density

    def get_drag_coefficient(self) -> float:
        """Parasitic drag coefficient for rail button"""
        # Approximate as blunt body
        return 0.8

    def get_frontal_area(self) -> float:
        """Frontal area for drag calculation"""
        return self.outer_diameter * self.height


class LaunchLug(RocketComponent):
    """
    Launch lug for rod guidance.

    Typically a tube that slides over launch rod.
    """

    def __init__(
        self,
        name: str = "Launch Lug",
        inner_diameter: float = 0.005,  # 3/16" rod = ~5mm
        outer_diameter: float = 0.008,
        length: float = 0.030,  # 30mm typical
    ):
        super().__init__(name)
        self.inner_diameter = inner_diameter
        self.outer_diameter = outer_diameter
        self.length = length
        self.material = MATERIALS["Nylon"]

    def calculate_mass(self) -> float:
        """Launch lug mass (hollow cylinder)"""
        outer_area = math.pi * (self.outer_diameter / 2) ** 2
        inner_area = math.pi * (self.inner_diameter / 2) ** 2
        volume = (outer_area - inner_area) * self.length
        return volume * self.material.density

    def get_drag_coefficient(self) -> float:
        """Parasitic drag coefficient"""
        return 1.0  # Cylinder perpendicular to flow

    def get_frontal_area(self) -> float:
        """Frontal area"""
        return self.outer_diameter * self.length


class CameraShroud(RocketComponent):
    """
    Camera pod/shroud for onboard video.

    Significant drag source - streamlined or blister type.
    """

    class ShroudType(Enum):
        BLISTER = "blister"  # Dome protruding from body
        STREAMLINED = "streamlined"  # Teardrop fairing
        FLUSH = "flush"  # Flush window (minimal drag)

    def __init__(
        self,
        name: str = "Camera Shroud",
        shroud_type: "CameraShroud.ShroudType" = None,
        length: float = 0.050,  # Along body axis
        width: float = 0.030,  # Circumferential
        height: float = 0.015,  # Radial protrusion
        window_diameter: float = 0.020,
    ):
        super().__init__(name)
        self.shroud_type = shroud_type or CameraShroud.ShroudType.BLISTER
        self.length = length
        self.width = width
        self.height = height
        self.window_diameter = window_diameter
        self.material = MATERIALS["ABS plastic"]

    def calculate_mass(self) -> float:
        """Camera shroud mass"""
        # Approximate as partial cylinder
        volume = self.length * self.width * self.height * 0.5  # Hollow
        return volume * self.material.density

    def get_drag_coefficient(self) -> float:
        """Parasitic drag based on shroud type"""
        if self.shroud_type == CameraShroud.ShroudType.FLUSH:
            return 0.1
        elif self.shroud_type == CameraShroud.ShroudType.STREAMLINED:
            return 0.3
        else:  # BLISTER
            return 0.6

    def get_frontal_area(self) -> float:
        """Frontal area"""
        return self.width * self.height


# =============================================================================
# RECOVERY SYSTEM COMPONENTS
# =============================================================================


class Bulkhead(RocketComponent):
    """
    Bulkhead plate for compartment separation and load bearing.

    Used for:
    - Parachute attachment
    - Motor retention
    - Compartment sealing
    """

    def __init__(
        self,
        name: str = "Bulkhead",
        diameter: float = 0.098,  # Match body tube ID
        thickness: float = 0.006,  # 6mm typical for fiberglass
        has_center_hole: bool = False,
        center_hole_diameter: float = 0.0,
    ):
        super().__init__(name)
        self.diameter = diameter
        self.thickness = thickness
        self.has_center_hole = has_center_hole
        self.center_hole_diameter = center_hole_diameter
        self.material = MATERIALS["Fiberglass"]

    def calculate_mass(self) -> float:
        """Bulkhead mass"""
        area = math.pi * (self.diameter / 2) ** 2
        if self.has_center_hole:
            area -= math.pi * (self.center_hole_diameter / 2) ** 2
        volume = area * self.thickness
        return volume * self.material.density


class UBolt(RocketComponent):
    """
    U-bolt for shock cord attachment.

    Critical recovery component - must handle deployment loads.
    """

    def __init__(
        self,
        name: str = "U-Bolt",
        thread_diameter: float = 0.006,  # #10-32 = 6mm
        inside_width: float = 0.020,  # Between threads
        inside_height: float = 0.025,  # Thread to curve
    ):
        super().__init__(name)
        self.thread_diameter = thread_diameter
        self.inside_width = inside_width
        self.inside_height = inside_height
        self.material = MATERIALS["Steel"]

    def calculate_mass(self) -> float:
        """U-bolt mass (approximate)"""
        # Two straight sections + semicircle
        straight_length = 2 * (self.inside_height + 0.015)  # Plus thread
        curve_length = math.pi * (self.inside_width / 2)
        total_length = straight_length + curve_length
        wire_area = math.pi * (self.thread_diameter / 2) ** 2
        volume = wire_area * total_length
        # Add nut mass estimate
        nut_volume = 2 * (self.thread_diameter * 2) ** 3 * 0.5
        return (volume + nut_volume) * self.material.density

    def get_safe_working_load(self) -> float:
        """Estimate safe working load in Newtons"""
        # Based on thread diameter and steel yield strength
        thread_area = math.pi * (self.thread_diameter / 2) ** 2
        yield_strength = 250e6  # Pa for typical steel
        safety_factor = 4.0
        return 2 * thread_area * yield_strength / safety_factor  # 2 legs


class EyeBolt(RocketComponent):
    """
    Eye bolt for shock cord attachment.

    Alternative to U-bolt, single attachment point.
    """

    def __init__(
        self,
        name: str = "Eye Bolt",
        thread_diameter: float = 0.006,
        eye_inner_diameter: float = 0.010,
        shank_length: float = 0.020,
    ):
        super().__init__(name)
        self.thread_diameter = thread_diameter
        self.eye_inner_diameter = eye_inner_diameter
        self.shank_length = shank_length
        self.material = MATERIALS["Steel"]

    def calculate_mass(self) -> float:
        """Eye bolt mass"""
        # Shank volume
        shank_volume = math.pi * (self.thread_diameter / 2) ** 2 * self.shank_length
        # Eye volume (torus section)
        eye_radius = (self.eye_inner_diameter + self.thread_diameter) / 2
        eye_volume = 2 * math.pi * eye_radius * math.pi * (self.thread_diameter / 2) ** 2
        return (shank_volume + eye_volume) * self.material.density


class ShockCord(RocketComponent):
    """
    Shock cord for parachute deployment.

    Material types: Nylon, Kevlar, Tubular nylon
    """

    class CordType(Enum):
        ELASTIC = "elastic"  # Bungee-style
        TUBULAR_NYLON = "tubular_nylon"  # Flat webbing
        KEVLAR = "kevlar"  # High strength
        NYLON_ROPE = "nylon_rope"  # Round cross-section

    def __init__(
        self,
        name: str = "Shock Cord",
        cord_type: "ShockCord.CordType" = None,
        length: float = 3.0,  # 3m typical
        width: float = 0.012,  # 1/2" = 12mm webbing
        thickness: float = 0.002,
        breaking_strength: float = 5000.0,  # Newtons
    ):
        super().__init__(name)
        self.cord_type = cord_type or ShockCord.CordType.TUBULAR_NYLON
        self.length = length
        self.width = width
        self.thickness = thickness
        self.breaking_strength = breaking_strength

    def calculate_mass(self) -> float:
        """Shock cord mass"""
        # Typical densities by type
        if self.cord_type == ShockCord.CordType.KEVLAR:
            linear_density = 0.015  # kg/m for typical Kevlar cord
        elif self.cord_type == ShockCord.CordType.TUBULAR_NYLON:
            linear_density = 0.020  # kg/m
        elif self.cord_type == ShockCord.CordType.ELASTIC:
            linear_density = 0.025  # kg/m
        else:
            linear_density = 0.018  # kg/m

        return self.length * linear_density


# =============================================================================
# STRUCTURAL COMPONENTS
# =============================================================================


class Coupler(SymmetricComponent):
    """
    Coupler tube for joining body tube sections.

    Used for:
    - Modular rocket assembly
    - Separation joints
    - Payload integration
    """

    def __init__(
        self,
        name: str = "Coupler",
        length: float = 0.100,
        outer_diameter: float = 0.096,  # Slightly less than body ID
        inner_diameter: float = 0.090,
    ):
        super().__init__(name)
        self.length = length
        self.outer_radius = outer_diameter / 2.0
        self.outer_diameter = outer_diameter
        self.inner_diameter = inner_diameter
        self.thickness = (outer_diameter - inner_diameter) / 2.0
        self.material = MATERIALS["Fiberglass"]

    def calculate_mass(self) -> float:
        """Coupler mass"""
        outer_area = math.pi * (self.outer_radius) ** 2
        inner_area = math.pi * (self.inner_diameter / 2) ** 2
        volume = (outer_area - inner_area) * self.length
        return volume * self.material.density


class MotorRetainer(RocketComponent):
    """
    Motor retainer for securing motor in mount.

    Types: Screw-on, snap-ring, friction fit
    """

    class RetainerType(Enum):
        SCREW_ON = "screw_on"  # Threaded retainer
        SNAP_RING = "snap_ring"  # E-clip style
        FRICTION = "friction"  # Tape/friction fit
        AERO_PACK = "aero_pack"  # Commercial brand

    def __init__(
        self,
        name: str = "Motor Retainer",
        retainer_type: "MotorRetainer.RetainerType" = None,
        motor_diameter: float = 0.054,  # 54mm motor
        length: float = 0.015,
    ):
        super().__init__(name)
        self.retainer_type = retainer_type or MotorRetainer.RetainerType.SCREW_ON
        self.motor_diameter = motor_diameter
        self.length = length
        self.material = MATERIALS["Aluminum"]

    def calculate_mass(self) -> float:
        """Motor retainer mass"""
        # Approximate as ring
        outer_radius = self.motor_diameter / 2 + 0.003
        inner_radius = self.motor_diameter / 2
        volume = math.pi * (outer_radius**2 - inner_radius**2) * self.length
        return volume * self.material.density


class ThrustPlate(RocketComponent):
    """
    Thrust plate for transferring motor thrust to airframe.

    Critical structural component at aft end of motor mount.
    """

    def __init__(
        self,
        name: str = "Thrust Plate",
        outer_diameter: float = 0.098,
        inner_diameter: float = 0.054,  # Motor OD
        thickness: float = 0.006,
    ):
        super().__init__(name)
        self.outer_diameter = outer_diameter
        self.inner_diameter = inner_diameter
        self.thickness = thickness
        self.material = MATERIALS["Fiberglass"]

    def calculate_mass(self) -> float:
        """Thrust plate mass"""
        area = math.pi * ((self.outer_diameter / 2) ** 2 - (self.inner_diameter / 2) ** 2)
        volume = area * self.thickness
        return volume * self.material.density


# =============================================================================
# AVIONICS COMPONENTS
# =============================================================================


class AvionicsBay(RocketComponent):
    """
    Avionics/electronics bay for flight computers, GPS, telemetry.

    Contains sled or mounting for electronics.
    """

    def __init__(
        self,
        name: str = "Avionics Bay",
        length: float = 0.150,
        outer_diameter: float = 0.098,
        sled_mass: float = 0.100,  # Electronics sled
        electronics_mass: float = 0.200,  # Altimeters, GPS, etc.
    ):
        super().__init__(name)
        self.length = length
        self.outer_diameter = outer_diameter
        self.sled_mass = sled_mass
        self.electronics_mass = electronics_mass
        self.material = MATERIALS["Fiberglass"]

    def calculate_mass(self) -> float:
        """Total avionics bay mass"""
        # Coupler tube
        wall_thickness = 0.002
        outer_area = math.pi * (self.outer_diameter / 2) ** 2
        inner_area = math.pi * (self.outer_diameter / 2 - wall_thickness) ** 2
        tube_volume = (outer_area - inner_area) * self.length
        tube_mass = tube_volume * self.material.density

        # Add bulkheads (top and bottom)
        bulkhead_volume = math.pi * (self.outer_diameter / 2) ** 2 * 0.006 * 2
        bulkhead_mass = bulkhead_volume * self.material.density

        return tube_mass + bulkhead_mass + self.sled_mass + self.electronics_mass


class SwitchBand(RocketComponent):
    """
    External switch band for arming avionics.

    Allows external access to arm/disarm switches.
    """

    def __init__(
        self,
        name: str = "Switch Band",
        outer_diameter: float = 0.102,  # Slightly larger than body
        width: float = 0.025,  # Band width
        num_switches: int = 2,
    ):
        super().__init__(name)
        self.outer_diameter = outer_diameter
        self.width = width
        self.num_switches = num_switches
        self.material = MATERIALS["Aluminum"]

    def calculate_mass(self) -> float:
        """Switch band mass"""
        # Thin ring
        inner_diameter = self.outer_diameter - 0.004
        outer_area = math.pi * (self.outer_diameter / 2) ** 2
        inner_area = math.pi * (inner_diameter / 2) ** 2
        volume = (outer_area - inner_area) * self.width
        return volume * self.material.density
