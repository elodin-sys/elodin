"""
3D Rocket Visualizer

Interactive 3D visualization of rocket assembly showing:
- Component layout
- CG and CP markers
- Dimensions and stability
- Real-time assembly viewer
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from typing import List, Tuple
import math

from rocket_components import (
    Rocket, RocketComponent, NoseCone, BodyTube, FinSet, Parachute,
    NoseShape
)


class RocketVisualizer:
    """3D visualization of rocket assembly"""
    
    def __init__(self, rocket: Rocket):
        self.rocket = rocket
        self.fig = None
        self.ax = None
    
    def draw_nose_cone(self, nose: NoseCone, ax, color='lightblue'):
        """Draw nose cone as a cone"""
        # Create cone vertices
        n_segments = 20
        theta = np.linspace(0, 2*np.pi, n_segments)
        
        # Base circle
        r = nose.diameter / 2
        x_base = np.ones(n_segments) * (nose.position_x + nose.length)
        y_base = r * np.cos(theta)
        z_base = r * np.sin(theta)
        
        # Tip point
        x_tip = nose.position_x
        y_tip = 0
        z_tip = 0
        
        # Draw cone surface
        for i in range(n_segments - 1):
            verts = [
                [x_base[i], y_base[i], z_base[i]],
                [x_base[i+1], y_base[i+1], z_base[i+1]],
                [x_tip, y_tip, z_tip]
            ]
            poly = Poly3DCollection([verts], alpha=0.7, facecolor=color, edgecolor='darkblue')
            ax.add_collection3d(poly)
    
    def draw_body_tube(self, tube: BodyTube, ax, color='lightgray'):
        """Draw body tube as a cylinder"""
        n_segments = 20
        theta = np.linspace(0, 2*np.pi, n_segments)
        
        r = tube.outer_diameter / 2
        x_start = tube.position_x
        x_end = tube.position_x + tube.length
        
        # Generate cylinder surface
        for i in range(n_segments - 1):
            y1 = r * np.cos(theta[i])
            z1 = r * np.sin(theta[i])
            y2 = r * np.cos(theta[i+1])
            z2 = r * np.sin(theta[i+1])
            
            verts = [
                [x_start, y1, z1],
                [x_end, y1, z1],
                [x_end, y2, z2],
                [x_start, y2, z2]
            ]
            poly = Poly3DCollection([verts], alpha=0.6, facecolor=color, edgecolor='gray')
            ax.add_collection3d(poly)
    
    def draw_fins(self, fins: FinSet, ax, color='yellow'):
        """Draw fin set"""
        r_body = self.rocket.body_diameter / 2
        
        for i in range(fins.fin_count):
            angle = 2 * np.pi * i / fins.fin_count
            
            # Fin coordinates (in rocket frame, then rotated)
            # Root leading edge
            x_le = fins.position_x
            # Root trailing edge  
            x_te = x_le + fins.root_chord
            # Tip leading edge
            x_tip_le = x_le + fins.sweep_length
            # Tip trailing edge
            x_tip_te = x_tip_le + fins.tip_chord
            
            # Fin profile in 2D (xz plane before rotation)
            fin_x = [x_le, x_te, x_tip_te, x_tip_le]
            fin_r = [r_body, r_body, r_body + fins.semi_span, r_body + fins.semi_span]
            
            # Rotate around rocket axis
            fin_y = [r * np.cos(angle) for r in fin_r]
            fin_z = [r * np.sin(angle) for r in fin_r]
            
            verts = [[fin_x[j], fin_y[j], fin_z[j]] for j in range(4)]
            poly = Poly3DCollection([verts], alpha=0.8, facecolor=color, edgecolor='orange')
            ax.add_collection3d(poly)
    
    def plot_assembly(self, show_cg_cp=True, show_dimensions=True):
        """Plot complete rocket assembly"""
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Draw components
        for comp in self.rocket.components:
            if isinstance(comp, NoseCone):
                self.draw_nose_cone(comp, self.ax)
            elif isinstance(comp, BodyTube):
                self.draw_body_tube(comp, self.ax)
            elif isinstance(comp, FinSet):
                self.draw_fins(comp, self.ax)
        
        # Get mass and aero properties
        mass_props = self.rocket.get_total_mass_properties()
        aero_props = self.rocket.get_total_aerodynamic_properties(0.3)
        
        cg_x = mass_props.cg_x
        cp_x = aero_props.cp_x
        
        if show_cg_cp:
            # Draw CG marker (red sphere)
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            r_marker = self.rocket.body_diameter / 4
            x_cg = cg_x + r_marker * np.outer(np.cos(u), np.sin(v))
            y_cg = r_marker * np.outer(np.sin(u), np.sin(v))
            z_cg = r_marker * np.outer(np.ones(np.size(u)), np.cos(v))
            self.ax.plot_surface(x_cg, y_cg, z_cg, color='red', alpha=0.9, label='CG')
            
            # Draw CP marker (blue sphere)
            x_cp = cp_x + r_marker * np.outer(np.cos(u), np.sin(v))
            y_cp = r_marker * np.outer(np.sin(u), np.sin(v))
            z_cp = r_marker * np.outer(np.ones(np.size(u)), np.cos(v))
            self.ax.plot_surface(x_cp, y_cp, z_cp, color='blue', alpha=0.9, label='CP')
            
            # Draw stability margin line
            self.ax.plot([cg_x, cp_x], [0, 0], [0, 0], 'g--', linewidth=3, label='Static Margin')
        
        # Set labels and limits
        self.ax.set_xlabel('X - Longitudinal (m)', fontsize=10)
        self.ax.set_ylabel('Y (m)', fontsize=10)
        self.ax.set_zlabel('Z (m)', fontsize=10)
        
        # Set aspect ratio
        max_range = max(
            cp_x + 0.2,
            self.rocket.body_diameter * 2
        )
        self.ax.set_xlim([0, max_range])
        self.ax.set_ylim([-max_range/2, max_range/2])
        self.ax.set_zlim([-max_range/2, max_range/2])
        
        # Title with stats
        static_margin = self.rocket.get_static_margin()
        stability_str = "✓ STABLE" if static_margin > 1.0 else "⚠ UNSTABLE"
        
        self.ax.set_title(
            f'{self.rocket.name}\n'
            f'Mass: {mass_props.mass:.3f} kg | CG: {cg_x:.3f} m | CP: {cp_x:.3f} m\n'
            f'Static Margin: {static_margin:.2f} cal {stability_str}',
            fontsize=12, fontweight='bold'
        )
        
        # Add text annotations
        if show_dimensions:
            # Find rocket length
            max_x = 0
            for comp in self.rocket.components:
                if isinstance(comp, NoseCone):
                    max_x = max(max_x, comp.position_x + comp.length)
                elif isinstance(comp, BodyTube):
                    max_x = max(max_x, comp.position_x + comp.length)
                elif isinstance(comp, FinSet):
                    max_x = max(max_x, comp.position_x + comp.root_chord)
            
            self.ax.text2D(0.02, 0.98, 
                          f'Length: {max_x:.3f} m\n'
                          f'Diameter: {self.rocket.body_diameter*1000:.1f} mm\n'
                          f'CNα: {aero_props.cn_alpha:.2f} /rad',
                          transform=self.ax.transAxes,
                          fontsize=10,
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='CG (Center of Gravity)'),
            Patch(facecolor='blue', label='CP (Center of Pressure)'),
            Patch(facecolor='green', alpha=0.5, label='Static Margin')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        plt.tight_layout()
        return self.fig, self.ax
    
    def save(self, filename: str):
        """Save visualization to file"""
        if self.fig is None:
            self.plot_assembly()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Rocket visualization saved to {filename}")
    
    def show(self):
        """Display visualization"""
        if self.fig is None:
            self.plot_assembly()
        plt.show()


def visualize_rocket_comparison(rocket1: Rocket, rocket2: Rocket, 
                                title1: str = "Rocket 1", title2: str = "Rocket 2"):
    """Compare two rocket designs side by side"""
    fig = plt.figure(figsize=(16, 8))
    
    # First rocket
    ax1 = fig.add_subplot(121, projection='3d')
    vis1 = RocketVisualizer(rocket1)
    vis1.ax = ax1
    vis1.fig = fig
    
    for comp in rocket1.components:
        if isinstance(comp, NoseCone):
            vis1.draw_nose_cone(comp, ax1)
        elif isinstance(comp, BodyTube):
            vis1.draw_body_tube(comp, ax1)
        elif isinstance(comp, FinSet):
            vis1.draw_fins(comp, ax1)
    
    mass_props1 = rocket1.get_total_mass_properties()
    aero_props1 = rocket1.get_total_aerodynamic_properties(0.3)
    ax1.set_title(f'{title1}\nSM: {rocket1.get_static_margin():.2f} cal')
    
    # Second rocket
    ax2 = fig.add_subplot(122, projection='3d')
    vis2 = RocketVisualizer(rocket2)
    vis2.ax = ax2
    vis2.fig = fig
    
    for comp in rocket2.components:
        if isinstance(comp, NoseCone):
            vis2.draw_nose_cone(comp, ax2)
        elif isinstance(comp, BodyTube):
            vis2.draw_body_tube(comp, ax2)
        elif isinstance(comp, FinSet):
            vis2.draw_fins(comp, ax2)
    
    mass_props2 = rocket2.get_total_mass_properties()
    aero_props2 = rocket2.get_total_aerodynamic_properties(0.3)
    ax2.set_title(f'{title2}\nSM: {rocket2.get_static_margin():.2f} cal')
    
    plt.tight_layout()
    plt.show()


def demo_visualizer():
    """Demo the rocket visualizer"""
    from rocket_components import MATERIALS
    
    # Build example rocket
    rocket = Rocket("Demo Rocket - L1 Certification")
    
    rocket.add_component(NoseCone(
        "Ogive Nose Cone",
        NoseShape.OGIVE,
        length=0.15,
        diameter=0.054,
        thickness=0.003,
        material=MATERIALS["Fiberglass"]
    ))
    
    rocket.add_component(BodyTube(
        "Upper Body Tube",
        length=0.40,
        outer_diameter=0.054,
        thickness=0.002,
        material=MATERIALS["Blue Tube"],
        position_x=0.15
    ))
    
    rocket.add_component(BodyTube(
        "Lower Body Tube",
        length=0.20,
        outer_diameter=0.054,
        thickness=0.002,
        material=MATERIALS["Blue Tube"],
        position_x=0.55
    ))
    
    rocket.add_component(FinSet(
        "Trapezoidal Fins",
        fin_count=4,
        root_chord=0.12,
        tip_chord=0.06,
        semi_span=0.10,
        sweep_length=0.04,
        thickness=0.004,
        material=MATERIALS["Fiberglass"],
        position_x=0.63
    ))
    
    # Print summary
    rocket.print_summary()
    
    # Visualize
    vis = RocketVisualizer(rocket)
    vis.plot_assembly()
    vis.save('/home/kush-mahajan/elodin/examples/rocket-barrowman/rocket_assembly.png')
    vis.show()


if __name__ == "__main__":
    demo_visualizer()

