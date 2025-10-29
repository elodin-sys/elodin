"""
Component-Based Rocket Design System

This module implements OpenRocket-style component architecture where rockets
are built from individual components (nose cones, body tubes, fins, etc.).
Each component tracks its own mass, CG, inertia, and aerodynamic properties.

Design Philosophy:
- Components are hierarchical (tree structure)
- Parent components contain child components
- Mass/CG/Inertia aggregate from children to parents
- Real-time stability analysis as components are added/modified
"""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod
from enum import Enum


class Material:
    """Material properties"""
    def __init__(self, name: str, density: float):
        """
        Args:
            name: Material name
            density: Density in kg/m³
        """
        self.name = name
        self.density = density  # kg/m³


# Common materials (OpenRocket defaults)
MATERIALS = {
    "Balsa": Material("Balsa", 170.0),
    "Plywood": Material("Plywood", 640.0),
    "Cardboard": Material("Cardboard", 680.0),
    "Fiberglass": Material("Fiberglass", 1850.0),
    "Carbon Fiber": Material("Carbon Fiber", 1780.0),
    "PLA": Material("PLA", 1240.0),
    "ABS": Material("ABS", 1040.0),
    "Aluminum": Material("Aluminum", 2700.0),
    "Polystyrene": Material("Polystyrene", 1050.0),
    "Blue Tube": Material("Blue Tube", 1100.0),  # Common rocket tubing
}


class NoseShape(Enum):
    """Nose cone shape types"""
    CONICAL = "conical"
    OGIVE = "ogive"
    ELLIPTICAL = "elliptical"
    PARABOLIC = "parabolic"
    POWER_SERIES = "power_series"
    HAACK = "haack"


class FinShape(Enum):
    """Fin shape types"""
    TRAPEZOIDAL = "trapezoidal"
    ELLIPTICAL = "elliptical"
    FREE_FORM = "free_form"


@dataclass
class MassProperties:
    """Mass properties of a component or assembly"""
    mass: float = 0.0  # kg
    cg_x: float = 0.0  # m from nose tip
    cg_y: float = 0.0  # m (typically 0 for axisymmetric)
    cg_z: float = 0.0  # m (typically 0 for axisymmetric)
    
    # Moments of inertia about CG (kg·m²)
    ixx: float = 0.0  # Roll (about longitudinal axis)
    iyy: float = 0.0  # Pitch
    izz: float = 0.0  # Yaw
    
    def __add__(self, other: 'MassProperties') -> 'MassProperties':
        """Combine two mass properties using parallel axis theorem"""
        if self.mass == 0:
            return other
        if other.mass == 0:
            return self
        
        total_mass = self.mass + other.mass
        
        # Combined CG using weighted average
        cg_x = (self.mass * self.cg_x + other.mass * other.cg_x) / total_mass
        cg_y = (self.mass * self.cg_y + other.mass * other.cg_y) / total_mass
        cg_z = (self.mass * self.cg_z + other.mass * other.cg_z) / total_mass
        
        # Distance from old CGs to new combined CG
        dx1 = self.cg_x - cg_x
        dy1 = self.cg_y - cg_y
        dz1 = self.cg_z - cg_z
        
        dx2 = other.cg_x - cg_x
        dy2 = other.cg_y - cg_y
        dz2 = other.cg_z - cg_z
        
        # Parallel axis theorem: I_combined = I1 + m1*d1² + I2 + m2*d2²
        ixx = (self.ixx + self.mass * (dy1**2 + dz1**2) +
               other.ixx + other.mass * (dy2**2 + dz2**2))
        
        iyy = (self.iyy + self.mass * (dx1**2 + dz1**2) +
               other.iyy + other.mass * (dx2**2 + dz2**2))
        
        izz = (self.izz + self.mass * (dx1**2 + dy1**2) +
               other.izz + other.mass * (dx2**2 + dy2**2))
        
        return MassProperties(
            mass=total_mass,
            cg_x=cg_x, cg_y=cg_y, cg_z=cg_z,
            ixx=ixx, iyy=iyy, izz=izz
        )


@dataclass
class AerodynamicProperties:
    """Aerodynamic properties of a component"""
    cn_alpha: float = 0.0  # Normal force coefficient slope (per radian)
    cp_x: float = 0.0  # Center of pressure from nose tip (m)
    cd_base: float = 0.0  # Base drag coefficient
    cd_friction: float = 0.0  # Skin friction drag coefficient


class RocketComponent(ABC):
    """Base class for all rocket components"""
    
    def __init__(self, name: str, position_x: float = 0.0):
        """
        Args:
            name: Component name
            position_x: Position from nose tip (m)
        """
        self.name = name
        self.position_x = position_x  # Position from nose tip
        self.parent: Optional['RocketComponent'] = None
        self.children: List['RocketComponent'] = []
        
        # Cached properties (invalidated when component changes)
        self._mass_props_cache: Optional[MassProperties] = None
        self._aero_props_cache: Optional[AerodynamicProperties] = None
    
    def add_child(self, child: 'RocketComponent'):
        """Add a child component"""
        child.parent = self
        self.children.append(child)
        self.invalidate_cache()
    
    def remove_child(self, child: 'RocketComponent'):
        """Remove a child component"""
        if child in self.children:
            self.children.remove(child)
            child.parent = None
            self.invalidate_cache()
    
    def invalidate_cache(self):
        """Invalidate cached properties (called when component changes)"""
        self._mass_props_cache = None
        self._aero_props_cache = None
        if self.parent:
            self.parent.invalidate_cache()
    
    @abstractmethod
    def calculate_own_mass_properties(self) -> MassProperties:
        """Calculate mass properties of this component only (no children)"""
        pass
    
    @abstractmethod
    def calculate_own_aerodynamic_properties(self, mach: float, body_diameter: float) -> AerodynamicProperties:
        """Calculate aerodynamic properties of this component only"""
        pass
    
    def get_mass_properties(self) -> MassProperties:
        """Get total mass properties (including children)"""
        if self._mass_props_cache is None:
            # Start with own properties
            props = self.calculate_own_mass_properties()
            
            # Add children
            for child in self.children:
                props = props + child.get_mass_properties()
            
            self._mass_props_cache = props
        
        return self._mass_props_cache
    
    def get_aerodynamic_properties(self, mach: float, body_diameter: float) -> AerodynamicProperties:
        """Get total aerodynamic properties (including children)"""
        # Start with own properties
        own_props = self.calculate_own_aerodynamic_properties(mach, body_diameter)
        
        # Combine with children (simplified - just sum CNα and weighted average CP)
        total_cn_alpha = own_props.cn_alpha
        weighted_cp = own_props.cn_alpha * own_props.cp_x
        total_cd = own_props.cd_base + own_props.cd_friction
        
        for child in self.children:
            child_props = child.get_aerodynamic_properties(mach, body_diameter)
            weighted_cp += child_props.cn_alpha * child_props.cp_x
            total_cn_alpha += child_props.cn_alpha
            total_cd += child_props.cd_base + child_props.cd_friction
        
        cp_x = weighted_cp / total_cn_alpha if total_cn_alpha > 1e-6 else own_props.cp_x
        
        return AerodynamicProperties(
            cn_alpha=total_cn_alpha,
            cp_x=cp_x,
            cd_base=total_cd,
            cd_friction=0.0
        )
    
    def get_all_components(self) -> List['RocketComponent']:
        """Get list of all components (self + all descendants)"""
        components = [self]
        for child in self.children:
            components.extend(child.get_all_components())
        return components


class NoseCone(RocketComponent):
    """Nose cone component"""
    
    def __init__(self, name: str, 
                 shape: NoseShape,
                 length: float,
                 diameter: float,
                 thickness: float,
                 material: Material,
                 position_x: float = 0.0):
        """
        Args:
            name: Component name
            shape: Nose cone shape
            length: Nose cone length (m)
            diameter: Base diameter (m)
            thickness: Wall thickness (m)
            material: Material
            position_x: Position from nose tip
        """
        super().__init__(name, position_x)
        self.shape = shape
        self.length = length
        self.diameter = diameter
        self.thickness = thickness
        self.material = material
    
    def calculate_own_mass_properties(self) -> MassProperties:
        """Calculate nose cone mass properties"""
        # Volume calculation depends on shape
        if self.shape == NoseShape.CONICAL:
            # Cone: V = (1/3)πr²h
            outer_volume = (1/3) * math.pi * (self.diameter/2)**2 * self.length
            inner_volume = (1/3) * math.pi * (self.diameter/2 - self.thickness)**2 * self.length
        elif self.shape == NoseShape.OGIVE:
            # Ogive approximation (simplified)
            outer_volume = 0.5 * math.pi * (self.diameter/2)**2 * self.length
            inner_volume = 0.5 * math.pi * (self.diameter/2 - self.thickness)**2 * self.length
        else:
            # Default to conical
            outer_volume = (1/3) * math.pi * (self.diameter/2)**2 * self.length
            inner_volume = (1/3) * math.pi * (self.diameter/2 - self.thickness)**2 * self.length
        
        volume = outer_volume - inner_volume
        mass = volume * self.material.density
        
        # CG location (from nose tip)
        # For cone: CG at 2/3 length from tip
        # For ogive: CG at ~0.5 length from tip
        if self.shape == NoseShape.CONICAL:
            cg_factor = 0.67
        elif self.shape == NoseShape.OGIVE:
            cg_factor = 0.466
        else:
            cg_factor = 0.6
        
        cg_x = self.position_x + cg_factor * self.length
        
        # Moment of inertia (simplified - thin shell approximation)
        r = self.diameter / 2
        ixx = 0.5 * mass * r**2  # Roll
        iyy = mass * (3*r**2 + self.length**2) / 12  # Pitch
        izz = iyy  # Yaw (same as pitch for axisymmetric)
        
        return MassProperties(
            mass=mass,
            cg_x=cg_x,
            ixx=ixx, iyy=iyy, izz=izz
        )
    
    def calculate_own_aerodynamic_properties(self, mach: float, body_diameter: float) -> AerodynamicProperties:
        """Calculate nose cone aerodynamics using Barrowman"""
        # Barrowman Equation 12: CNα_nose = 2 for slender bodies
        cn_alpha = 2.0
        
        # CP location (from nose tip)
        # Barrowman Equation 39
        if self.shape == NoseShape.CONICAL:
            cp_x = self.position_x + 0.67 * self.length
        elif self.shape == NoseShape.OGIVE:
            cp_x = self.position_x + 0.466 * self.length
        else:
            cp_x = self.position_x + 0.5 * self.length
        
        # Compressibility correction
        if mach > 0 and mach < 0.8:
            beta = math.sqrt(1.0 - mach**2)
            cn_alpha /= beta
        
        # Base drag (if nose has base area - typically zero for nose)
        cd_base = 0.0
        
        # Friction drag (simplified)
        wetted_area = math.pi * (self.diameter/2) * self.length  # Approximate
        reynolds = 1e6  # Assume turbulent
        cf = 0.074 / reynolds**0.2  # Turbulent flat plate
        cd_friction = cf * wetted_area / (math.pi * (body_diameter/2)**2)
        
        return AerodynamicProperties(
            cn_alpha=cn_alpha,
            cp_x=cp_x,
            cd_base=cd_base,
            cd_friction=cd_friction
        )


class BodyTube(RocketComponent):
    """Body tube component"""
    
    def __init__(self, name: str,
                 length: float,
                 outer_diameter: float,
                 thickness: float,
                 material: Material,
                 position_x: float = 0.0):
        """
        Args:
            name: Component name
            length: Tube length (m)
            outer_diameter: Outer diameter (m)
            thickness: Wall thickness (m)
            material: Material
            position_x: Position from nose tip
        """
        super().__init__(name, position_x)
        self.length = length
        self.outer_diameter = outer_diameter
        self.thickness = thickness
        self.material = material
    
    def calculate_own_mass_properties(self) -> MassProperties:
        """Calculate body tube mass properties"""
        # Thin-walled cylinder
        outer_r = self.outer_diameter / 2
        inner_r = outer_r - self.thickness
        
        # Volume
        volume = math.pi * (outer_r**2 - inner_r**2) * self.length
        mass = volume * self.material.density
        
        # CG at geometric center
        cg_x = self.position_x + self.length / 2
        
        # Moments of inertia
        r_mean = (outer_r + inner_r) / 2
        ixx = mass * r_mean**2  # Roll (thin ring)
        iyy = mass * (3*r_mean**2 + self.length**2) / 12  # Pitch
        izz = iyy
        
        return MassProperties(
            mass=mass,
            cg_x=cg_x,
            ixx=ixx, iyy=iyy, izz=izz
        )
    
    def calculate_own_aerodynamic_properties(self, mach: float, body_diameter: float) -> AerodynamicProperties:
        """Body tube has minimal direct aerodynamic contribution"""
        # Body tube itself doesn't generate significant normal force
        cn_alpha = 0.0
        cp_x = self.position_x + self.length / 2
        
        # Friction drag
        wetted_area = math.pi * self.outer_diameter * self.length
        reynolds = 1e6
        cf = 0.074 / reynolds**0.2
        cd_friction = cf * wetted_area / (math.pi * (body_diameter/2)**2)
        
        return AerodynamicProperties(
            cn_alpha=cn_alpha,
            cp_x=cp_x,
            cd_base=0.0,
            cd_friction=cd_friction
        )


class FinSet(RocketComponent):
    """Fin set component (trapezoidal fins)"""
    
    def __init__(self, name: str,
                 fin_count: int,
                 root_chord: float,
                 tip_chord: float,
                 semi_span: float,
                 sweep_length: float,
                 thickness: float,
                 material: Material,
                 position_x: float = 0.0):
        """
        Args:
            name: Component name
            fin_count: Number of fins (typically 3 or 4)
            root_chord: Root chord length (m)
            tip_chord: Tip chord length (m)
            semi_span: Fin height/span (m)
            sweep_length: Leading edge sweep distance (m)
            thickness: Fin thickness (m)
            material: Material
            position_x: Position of fin root leading edge from nose tip
        """
        super().__init__(name, position_x)
        self.fin_count = fin_count
        self.root_chord = root_chord
        self.tip_chord = tip_chord
        self.semi_span = semi_span
        self.sweep_length = sweep_length
        self.thickness = thickness
        self.material = material
    
    def calculate_own_mass_properties(self) -> MassProperties:
        """Calculate fin set mass properties"""
        # Single fin area (trapezoidal)
        single_fin_area = 0.5 * (self.root_chord + self.tip_chord) * self.semi_span
        
        # Volume and mass
        single_fin_volume = single_fin_area * self.thickness
        single_fin_mass = single_fin_volume * self.material.density
        total_mass = single_fin_mass * self.fin_count
        
        # CG of single trapezoidal fin (from root leading edge)
        # x: along root chord
        # y: along span
        cg_x_local = (self.root_chord + 2*self.tip_chord) / (3*(self.root_chord + self.tip_chord)) * self.root_chord
        cg_y_local = self.semi_span / 3 * (self.root_chord + 2*self.tip_chord) / (self.root_chord + self.tip_chord)
        
        # CG in rocket coordinates (fins are symmetric, so y/z cancel out)
        cg_x = self.position_x + cg_x_local
        
        # Moment of inertia (simplified - treat as flat plate)
        # Single fin about its CG
        i_single_xx = single_fin_mass * self.semi_span**2 / 3  # About longitudinal axis
        i_single_yy = single_fin_mass * self.root_chord**2 / 12  # About pitch axis
        
        # Total for all fins (symmetric arrangement)
        ixx = i_single_xx * self.fin_count
        
        # Pitch/yaw inertia (parallel axis theorem)
        iyy = i_single_yy * self.fin_count + total_mass * (self.semi_span/2)**2
        izz = iyy
        
        return MassProperties(
            mass=total_mass,
            cg_x=cg_x,
            ixx=ixx, iyy=iyy, izz=izz
        )
    
    def calculate_own_aerodynamic_properties(self, mach: float, body_diameter: float) -> AerodynamicProperties:
        """Calculate fin aerodynamics using Barrowman"""
        # Barrowman fin theory
        # Single fin CNα (Equation 57)
        d = body_diameter
        s = self.semi_span
        cr = self.root_chord
        ct = self.tip_chord
        
        # Mid-chord line length
        mid_chord_sweep = self.sweep_length / 2
        l = math.sqrt(s**2 + mid_chord_sweep**2)
        
        cn_single = 8 * (s/d)**2 / (1 + math.sqrt(1 + (2*l/(cr+ct))**2))
        
        # Fin set CNα (Equation 58)
        if self.fin_count == 4:
            cn_set = 2 * cn_single
        elif self.fin_count == 3:
            cn_set = math.sqrt(3) * cn_single
        else:
            cn_set = self.fin_count / 2 * cn_single
        
        # Body interference factor (Equation 77)
        r = d / 2
        k = 1.0 + r / (s + r)
        
        cn_alpha = k * cn_set
        
        # Compressibility correction
        if 0 < mach < 0.8:
            beta = math.sqrt(1.0 - mach**2)
            cn_alpha /= beta
        
        # CP location (Equation 76)
        term1 = (self.sweep_length / 3) * ((cr + 2*ct) / (cr + ct))
        term2 = (1/6) * (cr + ct - (cr*ct)/(cr+ct))
        cp_x = self.position_x + term1 + term2
        
        # Fin drag
        # Base drag at trailing edges
        trailing_edge_area = self.thickness * self.semi_span * self.fin_count
        ref_area = math.pi * (d/2)**2
        cd_base = 0.029 * trailing_edge_area / ref_area
        
        # Interference drag
        cd_interference = 0.0004 * self.fin_count
        
        return AerodynamicProperties(
            cn_alpha=cn_alpha,
            cp_x=cp_x,
            cd_base=cd_base + cd_interference,
            cd_friction=0.0
        )


class Parachute(RocketComponent):
    """Parachute recovery component"""
    
    def __init__(self, name: str,
                 diameter: float,
                 cd_parachute: float = 0.75,
                 deployment_altitude: float = None,
                 deployment_velocity: float = None,
                 deployment_time: float = None,
                 packed_mass: float = 0.05,
                 position_x: float = 0.0):
        """
        Args:
            name: Component name
            diameter: Parachute diameter when deployed (m)
            cd_parachute: Drag coefficient of parachute (typically 0.75-1.5)
            deployment_altitude: Deploy at this altitude (m), None = manual
            deployment_velocity: Deploy at this velocity (m/s), None = manual
            deployment_time: Deploy at this time (s), None = manual
            packed_mass: Mass of packed parachute (kg)
            position_x: Position from nose tip
        """
        super().__init__(name, position_x)
        self.diameter = diameter
        self.cd_parachute = cd_parachute
        self.deployment_altitude = deployment_altitude
        self.deployment_velocity = deployment_velocity
        self.deployment_time = deployment_time
        self.packed_mass = packed_mass
        self.deployed = False
    
    def calculate_own_mass_properties(self) -> MassProperties:
        """Parachute mass (minimal)"""
        return MassProperties(
            mass=self.packed_mass,
            cg_x=self.position_x,
            ixx=0.0, iyy=0.0, izz=0.0
        )
    
    def calculate_own_aerodynamic_properties(self, mach: float, body_diameter: float) -> AerodynamicProperties:
        """Parachute drag when deployed"""
        if self.deployed:
            # Deployed parachute drag
            parachute_area = math.pi * (self.diameter/2)**2
            ref_area = math.pi * (body_diameter/2)**2
            cd_total = self.cd_parachute * parachute_area / ref_area
            
            return AerodynamicProperties(
                cn_alpha=0.0,
                cp_x=self.position_x,
                cd_base=cd_total,
                cd_friction=0.0
            )
        else:
            # Packed parachute (negligible drag)
            return AerodynamicProperties(
                cn_alpha=0.0,
                cp_x=self.position_x,
                cd_base=0.0,
                cd_friction=0.0
            )
    
    def check_deployment(self, time: float, altitude: float, velocity_vertical: float) -> bool:
        """
        Check if parachute should deploy
        
        Args:
            time: Current time (s)
            altitude: Current altitude (m)
            velocity_vertical: Vertical velocity (m/s, negative = descending)
        """
        if self.deployed:
            return False
        
        # Don't deploy during ascent (vy > 0) or early in flight
        if time < 1.0 or velocity_vertical > -0.1:
            return False
        
        if self.deployment_time is not None and time >= self.deployment_time:
            return True
        
        # Deploy when descending below target altitude
        if self.deployment_altitude is not None and altitude <= self.deployment_altitude:
            return True
        
        # Deploy when descending below target velocity
        if self.deployment_velocity is not None and velocity_vertical <= -self.deployment_velocity:
            return True
        
        return False


class Rocket:
    """Complete rocket assembly"""
    
    def __init__(self, name: str):
        self.name = name
        self.components: List[RocketComponent] = []
        self.body_diameter: float = 0.0  # Will be set from body tube
        
    def add_component(self, component: RocketComponent):
        """Add a component to the rocket"""
        self.components.append(component)
        
        # Update body diameter if this is a body tube
        if isinstance(component, BodyTube):
            self.body_diameter = max(self.body_diameter, component.outer_diameter)
        elif isinstance(component, NoseCone):
            self.body_diameter = max(self.body_diameter, component.diameter)
    
    def get_total_mass_properties(self) -> MassProperties:
        """Get total mass properties of entire rocket"""
        total = MassProperties()
        for component in self.components:
            total = total + component.get_mass_properties()
        return total
    
    def get_total_aerodynamic_properties(self, mach: float = 0.3) -> AerodynamicProperties:
        """Get total aerodynamic properties"""
        total_cn_alpha = 0.0
        weighted_cp = 0.0
        total_cd = 0.0
        
        for component in self.components:
            aero = component.get_aerodynamic_properties(mach, self.body_diameter)
            weighted_cp += aero.cn_alpha * aero.cp_x
            total_cn_alpha += aero.cn_alpha
            total_cd += aero.cd_base + aero.cd_friction
        
        cp_x = weighted_cp / total_cn_alpha if total_cn_alpha > 1e-6 else 0.0
        
        return AerodynamicProperties(
            cn_alpha=total_cn_alpha,
            cp_x=cp_x,
            cd_base=total_cd,
            cd_friction=0.0
        )
    
    def get_static_margin(self, mach: float = 0.3) -> float:
        """
        Calculate static margin (stability)
        
        Returns:
            Static margin in calibers (CP - CG) / body_diameter
            Positive = stable, negative = unstable
            Recommended: 1-2 calibers
        """
        mass_props = self.get_total_mass_properties()
        aero_props = self.get_total_aerodynamic_properties(mach)
        
        cg = mass_props.cg_x
        cp = aero_props.cp_x
        
        if self.body_diameter == 0:
            return 0.0
        
        return (cp - cg) / self.body_diameter
    
    def print_summary(self):
        """Print rocket summary"""
        print(f"\n{'='*60}")
        print(f"ROCKET: {self.name}")
        print(f"{'='*60}")
        
        # Components
        print(f"\nComponents ({len(self.components)}):")
        for i, comp in enumerate(self.components, 1):
            print(f"  {i}. {comp.name} @ x={comp.position_x:.3f}m")
        
        # Mass properties
        mass_props = self.get_total_mass_properties()
        print(f"\nMass Properties:")
        print(f"  Total mass: {mass_props.mass:.3f} kg")
        print(f"  CG location: {mass_props.cg_x:.3f} m from nose tip")
        print(f"  Ixx (roll): {mass_props.ixx:.6f} kg·m²")
        print(f"  Iyy (pitch): {mass_props.iyy:.6f} kg·m²")
        print(f"  Izz (yaw): {mass_props.izz:.6f} kg·m²")
        
        # Aerodynamic properties
        aero_props = self.get_total_aerodynamic_properties(mach=0.3)
        print(f"\nAerodynamic Properties (Mach 0.3):")
        print(f"  CNα: {aero_props.cn_alpha:.3f} /rad")
        print(f"  CP location: {aero_props.cp_x:.3f} m from nose tip")
        print(f"  CD (total): {aero_props.cd_base + aero_props.cd_friction:.3f}")
        
        # Stability
        static_margin = self.get_static_margin(mach=0.3)
        print(f"\nStability:")
        print(f"  Static margin: {static_margin:.2f} calibers")
        if static_margin < 0:
            print(f"  ⚠️  UNSTABLE - CP is ahead of CG!")
        elif static_margin < 1.0:
            print(f"  ⚠️  MARGINALLY STABLE - Increase fin size or move CG forward")
        elif static_margin > 3.0:
            print(f"  ⚠️  OVERSTABLE - May be difficult to turn, consider smaller fins")
        else:
            print(f"  ✓ STABLE - Good design!")
        
        print(f"{'='*60}\n")


def example_rocket():
    """Build an example rocket similar to an Estes Alpha"""
    rocket = Rocket("Alpha-Style Model Rocket")
    
    # Nose cone
    nose = NoseCone(
        name="Nose Cone",
        shape=NoseShape.OGIVE,
        length=0.07,  # 70mm
        diameter=0.024,  # BT-50 (24mm)
        thickness=0.002,  # 2mm wall
        material=MATERIALS["Balsa"],
        position_x=0.0
    )
    rocket.add_component(nose)
    
    # Body tube
    body = BodyTube(
        name="Body Tube",
        length=0.300,  # 300mm
        outer_diameter=0.024,  # BT-50
        thickness=0.0008,  # 0.8mm cardboard
        material=MATERIALS["Cardboard"],
        position_x=0.07
    )
    rocket.add_component(body)
    
    # Fin set
    fins = FinSet(
        name="Fin Set",
        fin_count=3,
        root_chord=0.06,  # 60mm
        tip_chord=0.03,  # 30mm
        semi_span=0.05,  # 50mm
        sweep_length=0.02,  # 20mm
        thickness=0.003,  # 3mm
        material=MATERIALS["Balsa"],
        position_x=0.300  # At rear of body
    )
    rocket.add_component(fins)
    
    # Parachute
    chute = Parachute(
        name="Main Parachute",
        diameter=0.30,  # 300mm
        cd_parachute=0.75,
        deployment_altitude=150.0,  # Deploy at 150m
        packed_mass=0.015,  # 15g
        position_x=0.05
    )
    rocket.add_component(chute)
    
    return rocket


def main():
    """Demo rocket builder"""
    rocket = example_rocket()
    rocket.print_summary()


if __name__ == "__main__":
    main()

