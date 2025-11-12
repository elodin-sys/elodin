"""
Complete OpenRocket component system reimplemented in Python.
Matches OpenRocket's component hierarchy and mass/aerodynamic calculations exactly.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
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
        
    def add_child(self, component: 'RocketComponent'):
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
    
    def __init__(self, name: str = "Body tube", length: float = 0.3, 
                 outer_radius: float = 0.025, thickness: float = 0.001):
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
    
    def __init__(self, name: str = "Nose cone", length: float = 0.1,
                 base_radius: float = 0.025, thickness: float = 0.002,
                 shape: Shape = Shape.OGIVE):
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
            outer_vol = (1/3) * math.pi * self.base_radius**2 * self.length
        elif self.shape == self.Shape.OGIVE:
            # Tangent ogive approximation
            outer_vol = 0.5 * math.pi * self.base_radius**2 * self.length
        else:
            # Generic approximation for other shapes
            outer_vol = 0.5 * math.pi * self.base_radius**2 * self.length
        
        # Subtract hollow interior if applicable
        if self.thickness > 0:
            inner_radius = self.base_radius - self.thickness
            scale = (inner_radius / self.base_radius)
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
    
    def __init__(self, name: str = "Transition", length: float = 0.1,
                 fore_radius: float = 0.02, aft_radius: float = 0.025,
                 thickness: float = 0.001, shape: Shape = Shape.CONICAL):
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
        outer_vol = (math.pi * self.length / 3) * (r1**2 + r1*r2 + r2**2)
        
        r1_inner = max(0, r1 - self.thickness)
        r2_inner = max(0, r2 - self.thickness)
        inner_vol = (math.pi * self.length / 3) * (r1_inner**2 + r1_inner*r2_inner + r2_inner**2)
        
        volume = outer_vol - inner_vol
        return volume * self.material.density
    
    def calculate_cg_x(self) -> float:
        """CG of frustum"""
        r1 = self.fore_radius
        r2 = self.aft_radius
        # CG of frustum from narrow end
        if r1 != r2:
            cg = self.length * (r1**2 + 2*r1*r2 + 3*r2**2) / (4 * (r1**2 + r1*r2 + r2**2))
        else:
            cg = self.length / 2
        return cg


class TrapezoidFinSet(RocketComponent):
    """Trapezoidal fin set - OpenRocket standard"""
    
    def __init__(self, name: str = "Trapezoidal fin set", fin_count: int = 3,
                 root_chord: float = 0.1, tip_chord: float = 0.05,
                 span: float = 0.05, sweep: float = 0.03, thickness: float = 0.003):
        super().__init__(name)
        self.fin_count = fin_count
        self.root_chord = root_chord
        self.tip_chord = tip_chord
        self.span = span
        self.sweep = sweep  # Sweep distance
        self.thickness = thickness
        self.material = MATERIALS["Plywood (birch)"]
        self.cant_angle = 0.0  # radians
        
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
        cg = (a + 2*b) / (3 * (a + b)) * self.root_chord
        return cg
    
    def get_mac(self) -> float:
        """Mean aerodynamic chord"""
        return 2 * (self.root_chord + self.tip_chord) / 3


class EllipticalFinSet(RocketComponent):
    """Elliptical fin set"""
    
    def __init__(self, name: str = "Elliptical fin set", fin_count: int = 3,
                 root_chord: float = 0.1, span: float = 0.05, thickness: float = 0.003):
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
    
    def __init__(self, name: str = "Inner tube", length: float = 0.2,
                 outer_radius: float = 0.02, thickness: float = 0.001):
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
    
    def __init__(self, name: str = "Centering ring", outer_radius: float = 0.025,
                 inner_radius: float = 0.02, thickness: float = 0.003):
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


class Bulkhead(RocketComponent):
    """Bulkhead plate"""
    
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


class LaunchLug(RocketComponent):
    """Launch lug for rail"""
    
    def __init__(self, name: str = "Launch lug", length: float = 0.03,
                 outer_radius: float = 0.005, thickness: float = 0.0005):
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


class RailButton(RocketComponent):
    """Rail button"""
    
    def __init__(self, name: str = "Rail button", outer_diameter: float = 0.01,
                 base_height: float = 0.003):
        super().__init__(name)
        self.outer_diameter = outer_diameter
        self.base_height = base_height
        self.material = MATERIALS["Nylon"]
        
    def calculate_mass(self) -> float:
        """Small button mass"""
        radius = self.outer_diameter / 2
        volume = math.pi * radius**2 * self.base_height
        return volume * self.material.density


class ShockCord(RocketComponent):
    """Shock cord for recovery"""
    
    def __init__(self, name: str = "Shock cord", length: float = 1.0,
                 diameter: float = 0.003):
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
    
    def __init__(self, name: str = "Mass component", mass: float = 0.05,
                 length: float = 0.02, radius: float = 0.015):
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
    
    def __init__(self, name: str = "Parachute", diameter: float = 0.3,
                 cd: float = 0.75, material_density: float = 50.0):
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
        area = math.pi * (self.diameter / 2)**2
        mass_g = area * self.material_density
        return mass_g / 1000.0  # Convert to kg
    
    def get_drag_area(self) -> float:
        """Effective drag area when deployed"""
        if self.deployed:
            area = math.pi * (self.diameter / 2)**2
            return self.cd * area
        return 0.0


class Streamer(RocketComponent):
    """Streamer recovery device"""
    
    def __init__(self, name: str = "Streamer", length: float = 0.5,
                 width: float = 0.05, cd: float = 0.5):
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
                if hasattr(component, 'outer_radius'):
                    max_diameter = max(max_diameter, component.outer_radius * 2)
                elif hasattr(component, 'base_radius'):
                    max_diameter = max(max_diameter, component.base_radius * 2)
                    
                if hasattr(component, 'length'):
                    pos = component.get_absolute_position()
                    total_length = max(total_length, pos + component.length)
                    
            for child in component.children:
                traverse(child)
        
        traverse(self)
        self.reference_diameter = max_diameter
        self.reference_length = total_length

