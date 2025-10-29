"""
OpenRocket .ork File Exporter

Export our rocket designs to OpenRocket format for validation.
OpenRocket .ork files are gzipped XML files.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
import gzip
from typing import Dict
import os

from rocket_components import (
    Rocket, NoseCone, BodyTube, FinSet, Parachute,
    NoseShape, MATERIALS
)


class OpenRocketExporter:
    """Export rocket designs to OpenRocket .ork format"""
    
    def __init__(self, rocket: Rocket):
        self.rocket = rocket
    
    def nose_shape_to_ork(self, shape: NoseShape) -> str:
        """Convert our nose shape to OpenRocket enum"""
        mapping = {
            NoseShape.CONICAL: "CONICAL",
            NoseShape.OGIVE: "OGIVE",
            NoseShape.ELLIPTICAL: "ELLIPSOID",
            NoseShape.PARABOLIC: "PARABOLIC",
            NoseShape.POWER_SERIES: "POWER",
            NoseShape.HAACK: "HAACK"
        }
        return mapping.get(shape, "OGIVE")
    
    def export_to_ork(self, filename: str):
        """Export rocket to .ork file"""
        # Create XML structure
        root = ET.Element("openrocket")
        root.set("version", "1.9")
        root.set("creator", "Elodin Rocket Simulator")
        
        # Document info
        doc = ET.SubElement(root, "rocket")
        doc.set("name", self.rocket.name)
        
        # Add axial offset reference
        subassembly = ET.SubElement(doc, "subassembly")
        
        # Stage
        stage = ET.SubElement(subassembly, "stage")
        
        # Add components
        for comp in self.rocket.components:
            if isinstance(comp, NoseCone):
                self._add_nosecone(stage, comp)
            elif isinstance(comp, BodyTube):
                self._add_bodytube(stage, comp)
            elif isinstance(comp, FinSet):
                self._add_finset(stage, comp)
            elif isinstance(comp, Parachute):
                self._add_parachute(stage, comp)
        
        # Pretty print XML
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
        
        # Write gzipped XML
        with gzip.open(filename, 'wt', encoding='utf-8') as f:
            f.write(xml_str)
        
        print(f"Exported to {filename}")
        print(f"You can now open this file in OpenRocket for validation!")
    
    def _add_nosecone(self, parent, nose: NoseCone):
        """Add nose cone to XML"""
        nc = ET.SubElement(parent, "nosecone")
        
        ET.SubElement(nc, "name").text = nose.name
        ET.SubElement(nc, "finish").text = "normal"
        ET.SubElement(nc, "material").text = f"[{nose.material.name}]"
        ET.SubElement(nc, "length").text = str(nose.length)
        ET.SubElement(nc, "thickness").text = str(nose.thickness)
        ET.SubElement(nc, "shape").text = self.nose_shape_to_ork(nose.shape)
        ET.SubElement(nc, "shapeparameter").text = "0"
        ET.SubElement(nc, "aftradius").text = str(nose.diameter / 2)
        ET.SubElement(nc, "aftshoulderradius").text = str(nose.diameter / 2 - nose.thickness)
        ET.SubElement(nc, "aftshoulderlength").text = "0"
        ET.SubElement(nc, "aftshoulderthickness").text = str(nose.thickness)
    
    def _add_bodytube(self, parent, tube: BodyTube):
        """Add body tube to XML"""
        bt = ET.SubElement(parent, "bodytube")
        
        ET.SubElement(bt, "name").text = tube.name
        ET.SubElement(bt, "finish").text = "normal"
        ET.SubElement(bt, "material").text = f"[{tube.material.name}]"
        ET.SubElement(bt, "length").text = str(tube.length)
        ET.SubElement(bt, "thickness").text = str(tube.thickness)
        ET.SubElement(bt, "radius").text = str(tube.outer_diameter / 2)
    
    def _add_finset(self, parent, fins: FinSet):
        """Add fin set to XML"""
        fs = ET.SubElement(parent, "trapezoidfinset")
        
        ET.SubElement(fs, "name").text = fins.name
        ET.SubElement(fs, "fincount").text = str(fins.fin_count)
        ET.SubElement(fs, "material").text = f"[{fins.material.name}]"
        ET.SubElement(fs, "thickness").text = str(fins.thickness)
        ET.SubElement(fs, "crosssection").text = "SQUARE"
        ET.SubElement(fs, "cant").text = "0"
        ET.SubElement(fs, "tabheight").text = "0"
        ET.SubElement(fs, "tablength").text = "0"
        ET.SubElement(fs, "taboffset").text = "0"
        
        # Fin geometry
        ET.SubElement(fs, "height").text = str(fins.semi_span)
        ET.SubElement(fs, "rootchord").text = str(fins.root_chord)
        ET.SubElement(fs, "tipchord").text = str(fins.tip_chord)
        ET.SubElement(fs, "sweeplength").text = str(fins.sweep_length)
    
    def _add_parachute(self, parent, chute: Parachute):
        """Add parachute to XML"""
        pc = ET.SubElement(parent, "parachute")
        
        ET.SubElement(pc, "name").text = chute.name
        ET.SubElement(pc, "cd").text = str(chute.cd_parachute)
        ET.SubElement(pc, "diameter").text = str(chute.diameter)
        ET.SubElement(pc, "material").text = "[Ripstop nylon, 30 g/m²]"
        
        # Deployment
        if chute.deployment_altitude is not None:
            ET.SubElement(pc, "deployaltitude").text = str(chute.deployment_altitude)
        if chute.deployment_time is not None:
            ET.SubElement(pc, "deploydelay").text = str(chute.deployment_time)


def create_openrocket_validation_file():
    """Create a standard rocket design for OpenRocket validation"""
    from motor_database import create_sample_motors
    
    # Build a well-documented test rocket
    rocket = Rocket("Validation Rocket - Aerotech F50")
    
    rocket.add_component(NoseCone(
        "Ogive Nose",
        NoseShape.OGIVE,
        length=0.15,
        diameter=0.054,  # 54mm
        thickness=0.003,
        material=MATERIALS["Fiberglass"]
    ))
    
    rocket.add_component(BodyTube(
        "Body Tube",
        length=0.60,
        outer_diameter=0.054,
        thickness=0.002,
        material=MATERIALS["Blue Tube"],
        position_x=0.15
    ))
    
    rocket.add_component(FinSet(
        "Fins",
        fin_count=4,
        root_chord=0.12,
        tip_chord=0.06,
        semi_span=0.10,
        sweep_length=0.04,
        thickness=0.004,
        material=MATERIALS["Fiberglass"],
        position_x=0.65
    ))
    
    rocket.add_component(Parachute(
        "Main",
        diameter=0.60,
        cd_parachute=0.75,
        deployment_time=8.0,
        packed_mass=0.050,
        position_x=0.10
    ))
    
    # Print specs
    print("\n" + "="*70)
    print("OPENROCKET VALIDATION ROCKET SPECS")
    print("="*70)
    rocket.print_summary()
    
    # Export
    exporter = OpenRocketExporter(rocket)
    filename = '/home/kush-mahajan/elodin/examples/rocket-barrowman/validation_rocket.ork'
    exporter.export_to_ork(filename)
    
    print("\n" + "="*70)
    print("VALIDATION INSTRUCTIONS")
    print("="*70)
    print("1. Open OpenRocket (https://openrocket.info)")
    print("2. Load the file: validation_rocket.ork")
    print("3. Add motor: Aerotech F50-6T")
    print("4. Run simulation with these settings:")
    print("   - Launch angle: 90° (vertical)")
    print("   - Launch rail: 1.5m")
    print("   - Wind: 0 m/s")
    print("   - Temperature: 15°C / 288K")
    print("5. Compare results:")
    print("   - Expected apogee: ~250-280m")
    print("   - Expected max velocity: ~70 m/s")
    print("   - Static margin: ~2.5-3.0 calibers")
    print("="*70)
    
    return rocket


if __name__ == "__main__":
    create_openrocket_validation_file()

