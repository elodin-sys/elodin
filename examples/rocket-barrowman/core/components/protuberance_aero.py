"""
Protuberance Aerodynamics Module

Calculates drag contributions from external protuberances:
- Rail buttons
- Launch lugs  
- Camera shrouds
- Switch bands
- Antennas
- Vents

Based on:
- Hoerner, "Fluid-Dynamic Drag" (1965)
- ESDU Data Items for bluff body drag
- NASA SP-8055 "Nose Pressure Drag"
- OpenRocket methods
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum


class ProtuberanceType(Enum):
    """Types of external protuberances"""
    RAIL_BUTTON = "rail_button"
    LAUNCH_LUG = "launch_lug"
    CAMERA_BLISTER = "camera_blister"
    CAMERA_STREAMLINED = "camera_streamlined"
    CAMERA_FLUSH = "camera_flush"
    SWITCH_BAND = "switch_band"
    ANTENNA = "antenna"
    VENT = "vent"
    CUSTOM = "custom"


@dataclass
class ProtuberanceGeometry:
    """
    Geometry definition for a protuberance.
    
    All dimensions in meters.
    """
    protuberance_type: ProtuberanceType
    
    # Position on rocket
    axial_position: float      # Distance from nose tip (m)
    circumferential_angle: float = 0.0  # Angle around body (radians)
    
    # Dimensions (interpretation depends on type)
    height: float = 0.0        # Radial protrusion from body
    length: float = 0.0        # Axial dimension (along rocket)
    width: float = 0.0         # Circumferential dimension
    diameter: float = 0.0      # For cylindrical protuberances
    
    # Shape factors
    leading_edge_radius: float = 0.0   # 0 = sharp, > 0 = rounded
    trailing_edge_angle: float = 90.0  # degrees, 0 = streamlined, 90 = blunt
    
    # Material/surface
    surface_roughness: float = 0.0001  # m (for skin friction)
    
    def get_frontal_area(self) -> float:
        """Calculate frontal area presented to flow"""
        if self.protuberance_type == ProtuberanceType.RAIL_BUTTON:
            # Cylindrical button
            return self.diameter * self.height
        elif self.protuberance_type == ProtuberanceType.LAUNCH_LUG:
            # Tube perpendicular to flow
            return self.diameter * self.length
        elif self.protuberance_type in [ProtuberanceType.CAMERA_BLISTER, 
                                         ProtuberanceType.CAMERA_STREAMLINED,
                                         ProtuberanceType.CAMERA_FLUSH]:
            return self.width * self.height
        elif self.protuberance_type == ProtuberanceType.SWITCH_BAND:
            # Annular protrusion
            return self.width * self.height
        elif self.protuberance_type == ProtuberanceType.ANTENNA:
            return self.diameter * self.height
        elif self.protuberance_type == ProtuberanceType.VENT:
            return math.pi * (self.diameter / 2) ** 2
        else:
            return self.width * self.height
    
    def get_wetted_area(self) -> float:
        """Calculate wetted surface area for skin friction"""
        if self.protuberance_type == ProtuberanceType.RAIL_BUTTON:
            # Approximate as cylinder + hemisphere
            cyl_area = math.pi * self.diameter * self.height
            cap_area = 0.5 * math.pi * self.diameter ** 2
            return cyl_area + cap_area
        elif self.protuberance_type == ProtuberanceType.LAUNCH_LUG:
            # Tube surface
            return math.pi * self.diameter * self.length
        elif self.protuberance_type == ProtuberanceType.CAMERA_BLISTER:
            # Hemisphere-ish
            return 2 * self.length * self.width
        elif self.protuberance_type == ProtuberanceType.CAMERA_STREAMLINED:
            # Teardrop
            return 2.5 * self.length * self.width
        else:
            return 2 * self.length * self.width


class ProtuberanceAerodynamics:
    """
    Calculate aerodynamic drag for protuberances.
    
    Uses Hoerner's methods for bluff body drag with corrections for:
    - Reynolds number effects
    - Mach number effects  
    - Boundary layer state
    - Interference effects
    """
    
    def __init__(self, body_diameter: float, body_length: float):
        """
        Args:
            body_diameter: Reference body diameter (m)
            body_length: Total rocket length (m)
        """
        self.body_diameter = body_diameter
        self.body_length = body_length
        self.protuberances: List[ProtuberanceGeometry] = []
    
    def add_protuberance(self, protuberance: ProtuberanceGeometry):
        """Add a protuberance to the analysis"""
        self.protuberances.append(protuberance)
    
    def calculate_total_drag_coefficient(
        self,
        velocity: float,
        rho: float,
        mu: float,
        mach: float,
        reference_area: float,
    ) -> Tuple[float, List[dict]]:
        """
        Calculate total drag coefficient from all protuberances.
        
        Args:
            velocity: Freestream velocity (m/s)
            rho: Air density (kg/m³)
            mu: Dynamic viscosity (Pa·s)
            mach: Mach number
            reference_area: Rocket reference area (m²)
            
        Returns:
            (total_cd, breakdown) where breakdown is list of per-protuberance data
        """
        total_cd = 0.0
        breakdown = []
        
        # Sort protuberances by axial position for wake effects
        sorted_prots = sorted(self.protuberances, key=lambda p: p.axial_position)
        
        for i, prot in enumerate(sorted_prots):
            # Calculate individual CD
            cd_base = self._calculate_base_cd(prot, velocity, rho, mu, mach)
            
            # Apply boundary layer correction based on position
            bl_factor = self._boundary_layer_factor(prot.axial_position)
            
            # Apply interference from upstream protuberances
            interference_factor = self._interference_factor(prot, sorted_prots[:i], velocity)
            
            # Apply Mach number correction
            mach_factor = self._mach_correction(mach, prot)
            
            # Final CD for this protuberance
            cd_prot = cd_base * bl_factor * interference_factor * mach_factor
            
            # Convert to reference area basis
            frontal_area = prot.get_frontal_area()
            cd_ref = cd_prot * frontal_area / reference_area
            
            total_cd += cd_ref
            
            breakdown.append({
                "type": prot.protuberance_type.value,
                "position": prot.axial_position,
                "cd_base": cd_base,
                "bl_factor": bl_factor,
                "interference_factor": interference_factor,
                "mach_factor": mach_factor,
                "cd_final": cd_prot,
                "cd_reference": cd_ref,
                "frontal_area": frontal_area,
            })
        
        return total_cd, breakdown
    
    def _calculate_base_cd(
        self,
        prot: ProtuberanceGeometry,
        velocity: float,
        rho: float,
        mu: float,
        mach: float,
    ) -> float:
        """
        Calculate base drag coefficient for a protuberance.
        
        Based on Hoerner "Fluid-Dynamic Drag" and ESDU data.
        """
        # Reynolds number based on characteristic dimension
        if prot.protuberance_type == ProtuberanceType.RAIL_BUTTON:
            char_dim = prot.diameter
        elif prot.protuberance_type == ProtuberanceType.LAUNCH_LUG:
            char_dim = prot.diameter
        else:
            char_dim = max(prot.height, prot.width, prot.diameter, 0.01)
        
        Re = rho * velocity * char_dim / mu if mu > 0 else 1e6
        Re = max(Re, 100)  # Minimum Reynolds number
        
        if prot.protuberance_type == ProtuberanceType.RAIL_BUTTON:
            return self._rail_button_cd(prot, Re)
        elif prot.protuberance_type == ProtuberanceType.LAUNCH_LUG:
            return self._launch_lug_cd(prot, Re)
        elif prot.protuberance_type == ProtuberanceType.CAMERA_BLISTER:
            return self._camera_blister_cd(prot, Re)
        elif prot.protuberance_type == ProtuberanceType.CAMERA_STREAMLINED:
            return self._camera_streamlined_cd(prot, Re)
        elif prot.protuberance_type == ProtuberanceType.CAMERA_FLUSH:
            return self._camera_flush_cd(prot, Re)
        elif prot.protuberance_type == ProtuberanceType.SWITCH_BAND:
            return self._switch_band_cd(prot, Re)
        elif prot.protuberance_type == ProtuberanceType.ANTENNA:
            return self._antenna_cd(prot, Re)
        elif prot.protuberance_type == ProtuberanceType.VENT:
            return self._vent_cd(prot, Re)
        else:
            return self._generic_cd(prot, Re)
    
    def _rail_button_cd(self, prot: ProtuberanceGeometry, Re: float) -> float:
        """
        Rail button drag coefficient.
        
        Model: Cylinder with hemispherical cap on flat plate.
        Based on Hoerner Fig. 3-19 and ESDU 79015.
        
        CD depends on:
        - Height/diameter ratio
        - Reynolds number
        - Leading edge shape
        """
        h_d = prot.height / prot.diameter if prot.diameter > 0 else 1.0
        
        # Base CD for cylinder on flat plate (Hoerner)
        # Short cylinders (h/d < 1): CD ≈ 0.8-1.0
        # Tall cylinders (h/d > 2): CD ≈ 0.6-0.8
        if h_d < 0.5:
            cd_base = 1.1  # Very short - high drag
        elif h_d < 1.0:
            cd_base = 0.9 + 0.4 * (1.0 - h_d)
        elif h_d < 2.0:
            cd_base = 0.8 + 0.1 * (2.0 - h_d)
        else:
            cd_base = 0.7  # Tall cylinder, more streamlined
        
        # Reynolds number correction (subcritical to supercritical)
        if Re < 2e5:
            re_factor = 1.0  # Subcritical
        elif Re < 5e5:
            # Transition region
            re_factor = 1.0 - 0.3 * (Re - 2e5) / 3e5
        else:
            re_factor = 0.7  # Supercritical - lower drag
        
        # Leading edge correction
        if prot.leading_edge_radius > 0:
            r_d = prot.leading_edge_radius / prot.diameter
            le_factor = 1.0 - 0.3 * min(r_d, 0.5)  # Rounded LE reduces drag
        else:
            le_factor = 1.0
        
        return cd_base * re_factor * le_factor
    
    def _launch_lug_cd(self, prot: ProtuberanceGeometry, Re: float) -> float:
        """
        Launch lug drag coefficient.
        
        Model: Cylinder in crossflow (perpendicular to freestream).
        Based on Hoerner Chapter 3 "Resistance of Two-Dimensional Bodies".
        
        CD for infinite cylinder ≈ 1.2 at subcritical Re
        Finite length correction: CD_finite = CD_inf * (1 - k/L/D)
        """
        L_D = prot.length / prot.diameter if prot.diameter > 0 else 5.0
        
        # Base CD for infinite cylinder in crossflow
        if Re < 2e5:
            cd_inf = 1.2  # Subcritical
        elif Re < 5e5:
            cd_inf = 1.2 - 0.7 * (Re - 2e5) / 3e5  # Transition
        else:
            cd_inf = 0.5  # Supercritical
        
        # Finite length correction (end effects reduce drag)
        # From Hoerner Fig. 3-16
        if L_D < 5:
            end_factor = 0.9
        elif L_D < 10:
            end_factor = 0.95
        else:
            end_factor = 1.0
        
        # Gap effect (lug has clearance from body)
        # Flow under lug reduces effective blockage
        gap_factor = 0.85  # Typical for launch lugs
        
        return cd_inf * end_factor * gap_factor
    
    def _camera_blister_cd(self, prot: ProtuberanceGeometry, Re: float) -> float:
        """
        Camera blister drag coefficient.
        
        Model: Hemispherical or dome protrusion.
        Based on Hoerner "Bodies of Low Drag" and NACA reports.
        """
        # Height/width aspect ratio affects drag
        h_w = prot.height / prot.width if prot.width > 0 else 0.5
        
        # Hemisphere CD ≈ 0.4-0.5
        # Flatter blisters have lower CD
        if h_w < 0.3:
            cd_base = 0.3  # Low profile
        elif h_w < 0.5:
            cd_base = 0.4 + 0.3 * (h_w - 0.3) / 0.2
        else:
            cd_base = 0.55 + 0.3 * min(h_w - 0.5, 0.5)  # Tall blister
        
        # Surface roughness effect (camera windows)
        roughness_factor = 1.1  # Window discontinuity
        
        return cd_base * roughness_factor
    
    def _camera_streamlined_cd(self, prot: ProtuberanceGeometry, Re: float) -> float:
        """
        Streamlined camera fairing drag coefficient.
        
        Model: Teardrop or NACA-style fairing.
        """
        # Fineness ratio (length/height)
        L_H = prot.length / prot.height if prot.height > 0 else 3.0
        
        # Optimal fineness ratio ≈ 3-4 for minimum drag
        # CD decreases with increasing fineness up to ~4
        if L_H < 2:
            cd_base = 0.15 + 0.1 * (2 - L_H)  # Short fairing
        elif L_H < 4:
            cd_base = 0.10 + 0.025 * (4 - L_H)  # Good streamlining
        else:
            cd_base = 0.08 + 0.005 * (L_H - 4)  # Too long, skin friction dominates
        
        return cd_base
    
    def _camera_flush_cd(self, prot: ProtuberanceGeometry, Re: float) -> float:
        """
        Flush camera window drag coefficient.
        
        Model: Recessed or flush-mounted window.
        Minimal protrusion drag, mainly cavity drag.
        """
        # Flush windows have very low drag
        # Main contribution is step at window edge
        
        # Cavity depth effect
        if prot.height < 0.002:  # Truly flush
            cd_base = 0.02
        else:
            cd_base = 0.05 + 0.1 * prot.height / 0.01
        
        return cd_base
    
    def _switch_band_cd(self, prot: ProtuberanceGeometry, Re: float) -> float:
        """
        Switch band drag coefficient.
        
        Model: Annular step around body.
        Based on step drag correlations.
        """
        # Step height relative to boundary layer thickness
        # Approximate BL thickness: δ ≈ 0.37 * x / Re_x^0.2
        x = prot.axial_position
        Re_x = Re * x / max(prot.width, 0.01)
        delta_bl = 0.37 * x / (Re_x ** 0.2) if Re_x > 0 else 0.005
        
        h_delta = prot.height / delta_bl if delta_bl > 0 else 1.0
        
        # Step drag (Hoerner)
        if h_delta < 0.5:
            cd_base = 0.1 * h_delta  # Small step, proportional
        elif h_delta < 2:
            cd_base = 0.05 + 0.15 * h_delta  # Moderate step
        else:
            cd_base = 0.35  # Large step, separation dominated
        
        return cd_base
    
    def _antenna_cd(self, prot: ProtuberanceGeometry, Re: float) -> float:
        """
        Antenna drag coefficient.
        
        Model: Thin rod or blade.
        """
        # Antenna typically very thin (high L/D)
        L_D = prot.height / prot.diameter if prot.diameter > 0 else 20.0
        
        if L_D > 10:
            # Thin rod - high drag due to separation
            cd_base = 1.2
        else:
            # Thicker, lower CD
            cd_base = 0.8 + 0.04 * L_D
        
        return cd_base
    
    def _vent_cd(self, prot: ProtuberanceGeometry, Re: float) -> float:
        """
        Vent hole drag coefficient.
        
        Model: Circular hole in surface.
        """
        # Vent drag depends on whether air flows through
        # Static vent (no flow): cavity drag
        # Active vent: additional momentum drag
        
        cd_base = 0.5  # Typical for small vent
        
        return cd_base
    
    def _generic_cd(self, prot: ProtuberanceGeometry, Re: float) -> float:
        """
        Generic protuberance drag coefficient.
        
        Uses shape factor based on geometry.
        """
        # Estimate based on aspect ratios
        h_w = prot.height / prot.width if prot.width > 0 else 1.0
        
        # Blunter shapes have higher CD
        if h_w < 0.5:
            cd_base = 0.5  # Low profile
        elif h_w < 1.0:
            cd_base = 0.7
        else:
            cd_base = 0.9  # Tall
        
        # Trailing edge effect
        if prot.trailing_edge_angle < 30:
            te_factor = 0.6  # Streamlined
        elif prot.trailing_edge_angle < 60:
            te_factor = 0.8
        else:
            te_factor = 1.0  # Blunt
        
        return cd_base * te_factor
    
    def _boundary_layer_factor(self, axial_position: float) -> float:
        """
        Correction factor for boundary layer state at protuberance location.
        
        Forward positions: thinner BL, higher local velocity
        Aft positions: thicker BL, lower local velocity in BL
        """
        # Relative position along body
        x_rel = axial_position / self.body_length if self.body_length > 0 else 0.5
        
        # Forward positions see higher velocities
        # Aft positions are partially shielded by BL
        if x_rel < 0.2:
            return 1.1  # Forward - high velocity
        elif x_rel < 0.5:
            return 1.0  # Mid - nominal
        elif x_rel < 0.8:
            return 0.95  # Aft - slightly shielded
        else:
            return 0.9  # Far aft - in thick BL
    
    def _interference_factor(
        self,
        prot: ProtuberanceGeometry,
        upstream_prots: List[ProtuberanceGeometry],
        velocity: float,
    ) -> float:
        """
        Correction for interference from upstream protuberances.
        
        Downstream protuberances in the wake of upstream ones
        experience reduced drag due to lower local velocity.
        """
        if not upstream_prots:
            return 1.0
        
        interference = 1.0
        
        for upstream in upstream_prots:
            # Check if this protuberance is in the wake
            axial_dist = prot.axial_position - upstream.axial_position
            
            # Circumferential distance
            circ_dist = abs(prot.circumferential_angle - upstream.circumferential_angle)
            circ_dist = min(circ_dist, 2 * math.pi - circ_dist)
            
            # Wake width grows with distance
            wake_width = upstream.width + 0.1 * axial_dist
            
            # Check if in wake
            if circ_dist * self.body_diameter / 2 < wake_width:
                # In wake - reduced drag
                # Wake deficit decays with distance
                deficit = 0.3 * math.exp(-axial_dist / (5 * upstream.height))
                interference *= (1 - deficit)
        
        return max(interference, 0.5)  # Cap reduction at 50%
    
    def _mach_correction(self, mach: float, prot: ProtuberanceGeometry) -> float:
        """
        Mach number correction for compressibility effects.
        
        Subsonic: Prandtl-Glauert correction
        Transonic: Empirical increase
        Supersonic: Wave drag considerations
        """
        if mach < 0.3:
            return 1.0  # Incompressible
        elif mach < 0.8:
            # Prandtl-Glauert
            beta = math.sqrt(1 - mach ** 2)
            return 1.0 / beta
        elif mach < 1.2:
            # Transonic - significant increase
            return 1.5 + 1.0 * (mach - 0.8) / 0.4
        else:
            # Supersonic
            # Wave drag for blunt bodies
            return 1.2 + 0.3 * (mach - 1.0)


def create_standard_rail_button(
    position: float,
    size: str = "standard",
    angle: float = 0.0,
) -> ProtuberanceGeometry:
    """
    Create a standard rail button geometry.
    
    Args:
        position: Axial position from nose (m)
        size: "mini", "standard", "1010", "1515"
        angle: Circumferential angle (radians)
    """
    sizes = {
        "mini": {"diameter": 0.008, "height": 0.010},
        "standard": {"diameter": 0.010, "height": 0.015},
        "1010": {"diameter": 0.010, "height": 0.010},
        "1515": {"diameter": 0.015, "height": 0.015},
    }
    
    dims = sizes.get(size, sizes["standard"])
    
    return ProtuberanceGeometry(
        protuberance_type=ProtuberanceType.RAIL_BUTTON,
        axial_position=position,
        circumferential_angle=angle,
        height=dims["height"],
        diameter=dims["diameter"],
        leading_edge_radius=dims["diameter"] / 4,  # Rounded top
    )


def create_standard_launch_lug(
    position: float,
    rod_size: str = "3/16",
    length: float = 0.030,
    angle: float = 0.0,
) -> ProtuberanceGeometry:
    """
    Create a standard launch lug geometry.
    
    Args:
        position: Axial position from nose (m)
        rod_size: "1/8", "3/16", "1/4"
        length: Lug length (m)
        angle: Circumferential angle (radians)
    """
    rod_diameters = {
        "1/8": 0.0032,   # 3.2mm
        "3/16": 0.0048,  # 4.8mm
        "1/4": 0.0064,   # 6.4mm
    }
    
    rod_dia = rod_diameters.get(rod_size, rod_diameters["3/16"])
    outer_dia = rod_dia + 0.003  # 3mm wall thickness
    
    return ProtuberanceGeometry(
        protuberance_type=ProtuberanceType.LAUNCH_LUG,
        axial_position=position,
        circumferential_angle=angle,
        height=outer_dia,  # Radial protrusion
        length=length,
        diameter=outer_dia,
    )


def create_camera_shroud(
    position: float,
    shroud_type: str = "blister",
    length: float = 0.050,
    width: float = 0.030,
    height: float = 0.015,
    angle: float = 0.0,
) -> ProtuberanceGeometry:
    """
    Create a camera shroud geometry.
    
    Args:
        position: Axial position from nose (m)
        shroud_type: "blister", "streamlined", "flush"
        length: Along body axis (m)
        width: Circumferential (m)
        height: Radial protrusion (m)
        angle: Circumferential angle (radians)
    """
    type_map = {
        "blister": ProtuberanceType.CAMERA_BLISTER,
        "streamlined": ProtuberanceType.CAMERA_STREAMLINED,
        "flush": ProtuberanceType.CAMERA_FLUSH,
    }
    
    prot_type = type_map.get(shroud_type, ProtuberanceType.CAMERA_BLISTER)
    
    return ProtuberanceGeometry(
        protuberance_type=prot_type,
        axial_position=position,
        circumferential_angle=angle,
        height=height,
        length=length,
        width=width,
        trailing_edge_angle=30 if shroud_type == "streamlined" else 90,
    )

