"""
AI/Heuristic-based rocket builder from natural language requirements.
Uses OpenAI API for natural language understanding and heuristic algorithms for rocket design.
Comprehensive parameter support with robust error handling and spelling tolerance.
"""

import re
import os
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
from dataclasses import dataclass
import math
import json

if TYPE_CHECKING:
    from motor_scraper import MotorData

# Try to import OpenAI, but make it optional
try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None
    # OpenAI is optional - AI Builder will show a message in the UI if unavailable


@dataclass
class RocketRequirements:
    """Comprehensive parsed rocket requirements from natural language"""

    # Altitude and performance
    target_altitude_m: Optional[float] = None  # meters
    max_velocity_ms: Optional[float] = None  # m/s (max speed)
    max_acceleration_ms2: Optional[float] = None  # m/s² (max G-force)
    max_mach: Optional[float] = None  # Max Mach number

    # Payload
    payload_mass_kg: Optional[float] = None
    payload_size: Optional[str] = None  # e.g., "6U", "3U", "1U"
    payload_dimensions: Optional[Tuple[float, float, float]] = None  # (length, width, height) in m

    # Dimensions and constraints
    total_rocket_mass_kg: Optional[float] = None  # Total wet mass
    rocket_length_m: Optional[float] = None  # Total length
    diameter_constraint_m: Optional[float] = None  # Max diameter in meters
    length_constraint_m: Optional[float] = None  # Max length in meters
    min_diameter_m: Optional[float] = None  # Min diameter

    # Fins
    fin_count: Optional[int] = None  # Number of fins (3, 4, etc.)
    fin_size: Optional[str] = None  # "small", "medium", "large", "custom"
    fin_shape: Optional[str] = None  # "trapezoidal", "elliptical", "swept", etc.

    # Recovery
    recovery_method: Optional[str] = None  # "parachute", "dual_deploy", "streamer", "none"
    landing_distance_m: Optional[float] = None  # Max distance from launch site (meters)
    landing_distance_miles: Optional[float] = None  # Max distance in miles
    landing_distance_km: Optional[float] = None  # Max distance in km
    drogue_diameter_m: Optional[float] = None  # Drogue chute size
    main_chute_diameter_m: Optional[float] = None  # Main chute size
    deployment_altitude_m: Optional[float] = None  # Main chute deployment altitude

    # Motor
    motor_preference: Optional[str] = None  # Motor class like "M", "L", "K"
    motor_designation: Optional[str] = None  # Specific motor like "M1670"
    motor_manufacturer: Optional[str] = None  # Manufacturer preference

    # Materials and construction
    body_material: Optional[str] = None  # "Fiberglass", "Carbon fiber", "Cardboard", etc.
    fin_material: Optional[str] = None
    nose_material: Optional[str] = None

    # Budget and constraints
    budget_usd: Optional[float] = None  # Budget in dollars
    max_cost: Optional[float] = None

    # Launch conditions
    rail_length_m: Optional[float] = None  # Launch rail length
    launch_angle_deg: Optional[float] = None  # Launch angle from vertical
    wind_speed_ms: Optional[float] = None  # Expected wind speed

    # Safety and regulations
    max_altitude_agl: Optional[float] = None  # Max altitude above ground level
    certification_required: Optional[str] = None  # "NAR", "TRA", etc.

    # Additional preferences
    stability_margin: Optional[float] = None  # CP-CG margin (calibers)
    color_preference: Optional[str] = None  # Just for fun
    special_requirements: Optional[str] = None  # Free text for other requirements


class RocketDesigner:
    """Comprehensive heuristic-based rocket designer"""

    # Standard cubesat dimensions (meters)
    CUBESAT_SIZES = {
        "1U": (0.1, 0.1, 0.1135),  # 10x10x11.35 cm
        "2U": (0.1, 0.1, 0.227),  # 10x10x22.7 cm
        "3U": (0.1, 0.1, 0.3405),  # 10x10x34.05 cm
        "6U": (0.1, 0.2, 0.3405),  # 10x20x34.05 cm
        "12U": (0.2, 0.2, 0.3405),  # 20x20x34.05 cm
    }

    # Material densities (kg/m³)
    MATERIAL_DENSITIES = {
        "Fiberglass": 1850.0,
        "Carbon fiber": 1780.0,
        "Cardboard": 680.0,
        "Aluminum": 2700.0,
        "Kraft phenolic": 950.0,
        "Blue tube": 1040.0,
        "Plywood": 630.0,
        "Balsa": 170.0,
    }

    # Common misspellings and variations for fuzzy matching
    SPELLING_VARIATIONS = {
        "altitude": ["altitude", "alt", "height"],
        "payload": ["payload", "pay load", "load"],
        "parachute": ["parachute", "parashute", "chute", "paracute"],
        "recover": ["recover", "recovery", "land"],
        "diameter": ["diameter", "width", "size"],
        "length": ["length", "long", "tall"],
        "weight": ["weight", "weigh", "mass", "wight"],
        "fins": ["fins", "fin", "finns", "wings"],
        "motor": ["motor", "engine", "propellant", "propulsion"],
        "dual deploy": ["dual deploy", "dual-deploy", "dualdeploy", "two stage recovery"],
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
            print("Using OpenAI API for natural language processing")
        else:
            self.openai_client = None
            print(
                "Using regex-based parsing (install openai and set OPENAI_API_KEY for better results)"
            )

    def _normalize_text(self, text: str) -> str:
        """Normalize text for better parsing - handle common misspellings"""
        text_lower = text.lower()

        # Replace common misspellings
        for correct, variations in self.SPELLING_VARIATIONS.items():
            for variant in variations:
                if variant != correct and variant in text_lower:
                    # Use regex to replace whole words only
                    text_lower = re.sub(r"\b" + re.escape(variant) + r"\b", correct, text_lower)

        return text_lower

    def parse_requirements(self, text: str) -> RocketRequirements:
        """
        Parse natural language requirements into structured format.
        Uses OpenAI API if available, otherwise falls back to enhanced regex parsing.

        Examples:
        - "I want a rocket that goes to 10000 ft, carries a 6U payload that weighs 10 lbs"
        - "Build me a rocket for a 3U cubesat weighing 5 kg to reach 5000 meters"
        - "Rocket to 20000 ft with 2 kg payload, max diameter 6 inches, 4 fins, dual deploy recovery"
        - "I need a rocket that goes 10000 ft, weighs 15 kg total, lands within 1 mile"
        """
        if self.use_openai:
            return self._parse_with_openai(text)
        else:
            return self._parse_with_regex(text)

    def _parse_with_openai(self, text: str) -> RocketRequirements:
        """Parse requirements using OpenAI API with comprehensive prompt"""
        try:
            prompt = f"""Extract ALL rocket requirements from this text and return as JSON. Be very thorough and catch everything mentioned, including misspellings:
"{text}"

Return a JSON object with these fields (use null if not specified):
- target_altitude_m: altitude in meters (convert from feet/ft, km, miles if needed)
- max_velocity_ms: max velocity/speed in m/s (convert from mph, km/h if needed)
- max_acceleration_ms2: max acceleration/G-force in m/s² (convert from G if needed, 1G=9.81 m/s²)
- max_mach: max Mach number if specified
- payload_mass_kg: payload mass in kg (convert from lbs, pounds, grams if needed)
- payload_size: cubesat size like "6U", "3U", "1U" (or null)
- payload_dimensions: [length_m, width_m, height_m] or null
- total_rocket_mass_kg: total rocket weight/mass in kg (convert from lbs if needed)
- rocket_length_m: total rocket length in meters (convert from feet, inches if needed)
- diameter_constraint_m: max diameter in meters (convert from inches/in, cm, mm if needed)
- length_constraint_m: max length in meters (convert from feet/ft, inches if needed)
- min_diameter_m: min diameter in meters
- fin_count: number of fins (3, 4, 5, etc.) as integer
- fin_size: "small", "medium", "large", "custom" or null
- fin_shape: "trapezoidal", "elliptical", "swept", etc. or null
- recovery_method: "parachute", "dual_deploy", "streamer", "none" or null
- landing_distance_m: max landing distance in meters (convert from miles, km, feet if needed)
- landing_distance_miles: max distance in miles
- landing_distance_km: max distance in km
- drogue_diameter_m: drogue chute diameter in meters
- main_chute_diameter_m: main chute diameter in meters
- deployment_altitude_m: main chute deployment altitude in meters
- motor_preference: motor class like "M", "L", "K", "J" or null
- motor_designation: specific motor like "M1670" or null
- motor_manufacturer: manufacturer name or null
- body_material: "Fiberglass", "Carbon fiber", "Cardboard", etc. or null
- fin_material: material for fins or null
- nose_material: material for nose cone or null
- budget_usd: budget in dollars/USD
- max_cost: max cost in dollars
- rail_length_m: launch rail length in meters
- launch_angle_deg: launch angle from vertical in degrees
- wind_speed_ms: wind speed in m/s
- max_altitude_agl: max altitude above ground level in meters
- certification_required: "NAR", "TRA", etc. or null
- stability_margin: CP-CG margin in calibers (typically 1-2)
- color_preference: color if mentioned or null
- special_requirements: any other requirements as string or null

IMPORTANT:
- Handle misspellings (e.g., "recover" -> recovery, "alt" -> altitude)
- Convert all units correctly (feet to meters, lbs to kg, etc.)
- Extract numbers even with typos (e.g., "10kg" or "10 kg" both work)
- Be generous with interpretations (e.g., "land within 1 mile" -> landing_distance_miles: 1.0)
- Only include fields that are explicitly mentioned or can be reasonably inferred
- Return only valid JSON, no other text."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Using newer model
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert rocket design assistant. Extract ALL requirements from natural language, handle misspellings gracefully, and return comprehensive structured JSON. Be thorough and catch every detail.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1000,
            )

            result_text = response.choices[0].message.content.strip()
            # Remove markdown code blocks if present
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
                result_text = result_text.strip()

            parsed = json.loads(result_text)

            # Convert to RocketRequirements
            req = RocketRequirements()
            req.target_altitude_m = parsed.get("target_altitude_m")
            req.max_velocity_ms = parsed.get("max_velocity_ms")
            req.max_acceleration_ms2 = parsed.get("max_acceleration_ms2")
            req.max_mach = parsed.get("max_mach")
            req.payload_mass_kg = parsed.get("payload_mass_kg")
            req.payload_size = parsed.get("payload_size")
            if parsed.get("payload_dimensions"):
                req.payload_dimensions = tuple(parsed["payload_dimensions"])
            req.total_rocket_mass_kg = parsed.get("total_rocket_mass_kg")
            req.rocket_length_m = parsed.get("rocket_length_m")
            req.diameter_constraint_m = parsed.get("diameter_constraint_m")
            req.length_constraint_m = parsed.get("length_constraint_m")
            req.min_diameter_m = parsed.get("min_diameter_m")
            req.fin_count = parsed.get("fin_count")
            req.fin_size = parsed.get("fin_size")
            req.fin_shape = parsed.get("fin_shape")
            req.recovery_method = parsed.get("recovery_method")
            req.landing_distance_m = parsed.get("landing_distance_m")
            req.landing_distance_miles = parsed.get("landing_distance_miles")
            req.landing_distance_km = parsed.get("landing_distance_km")
            req.drogue_diameter_m = parsed.get("drogue_diameter_m")
            req.main_chute_diameter_m = parsed.get("main_chute_diameter_m")
            req.deployment_altitude_m = parsed.get("deployment_altitude_m")
            req.motor_preference = parsed.get("motor_preference")
            req.motor_designation = parsed.get("motor_designation")
            req.motor_manufacturer = parsed.get("motor_manufacturer")
            req.body_material = parsed.get("body_material")
            req.fin_material = parsed.get("fin_material")
            req.nose_material = parsed.get("nose_material")
            req.budget_usd = parsed.get("budget_usd") or parsed.get("budget")
            req.max_cost = parsed.get("max_cost")
            req.rail_length_m = parsed.get("rail_length_m")
            req.launch_angle_deg = parsed.get("launch_angle_deg")
            req.wind_speed_ms = parsed.get("wind_speed_ms")
            req.max_altitude_agl = parsed.get("max_altitude_agl")
            req.certification_required = parsed.get("certification_required")
            req.stability_margin = parsed.get("stability_margin")
            req.color_preference = parsed.get("color_preference")
            req.special_requirements = parsed.get("special_requirements")

            return req

        except Exception as e:
            print(f"Error using OpenAI API: {e}, falling back to regex parsing")
            return self._parse_with_regex(text)

    def _parse_with_regex(self, text: str) -> RocketRequirements:
        """Enhanced regex parsing with spelling tolerance"""
        req = RocketRequirements()
        text_normalized = self._normalize_text(text)

        # Extract altitude - many patterns with unit conversions
        # Try patterns with units first (more specific), then patterns without units
        altitude_patterns = [
            # Patterns with explicit units (most reliable)
            (r"(\d+\.?\d*)\s*(?:thousand|k)\s*ft", 304.8),  # "10k ft" = 10000 ft
            (
                r"(\d+\.?\d*)\s*ft\b",
                0.3048,
            ),  # feet (word boundary to avoid matching "ft" in other words)
            (r"(\d+\.?\d*)\s*feet", 0.3048),
            (r"(\d+\.?\d*)ft\b", 0.3048),  # "10000ft" without space
            (r"(\d+\.?\d*)\s*km\b", 1000.0),  # kilometers
            (r"(\d+\.?\d*)\s*kilometers?", 1000.0),
            (r"(\d+\.?\d*)\s*m\b(?!s)", 1.0),  # meters (not m/s)
            (r"(\d+\.?\d*)\s*meters?", 1.0),
            # Patterns with context words (no explicit unit - need heuristic)
            # Use None as multiplier to signal "apply heuristic"
            (r"goes?\s+to\s+(\d+\.?\d*)", None),  # No unit - apply heuristic
            (r"reach\s+(\d+\.?\d*)", None),
            (r"altitude\s+of\s+(\d+\.?\d*)", None),
            (r"(\d+\.?\d*)\s*(?:thousand|k)\s*(?:feet|ft)", 304.8),
        ]
        for pattern, multiplier in altitude_patterns:
            match = re.search(pattern, text_normalized)
            if match:
                value = float(match.group(1))
                if multiplier is None:
                    # Heuristic for patterns without explicit units:
                    # Large values (>1000) are likely meters, smaller values are likely feet
                    if value > 1000:
                        req.target_altitude_m = value  # Interpret as meters
                    else:
                        req.target_altitude_m = value * 0.3048  # Interpret as feet
                else:
                    # Apply the multiplier directly - each pattern has the correct conversion factor
                    req.target_altitude_m = value * multiplier
                break

        # Extract total rocket mass FIRST (higher priority)
        # "50 lb rocket" means total rocket mass, not payload
        total_mass_patterns = [
            (
                r"(\d+\.?\d*)\s*(?:lb|lbs|pound|pounds)\s+rocket",
                0.453592,
            ),  # "50 lb rocket" - HIGHEST PRIORITY
            (
                r"rocket\s+(?:that\s+)?(?:weighs?|mass|weight)\s+(\d+\.?\d*)",
                0.453592,
            ),  # "rocket weighs 50"
            (
                r"(\d+\.?\d*)\s*(?:lb|lbs|pound|pounds)\s+(?:rocket|total)",
                0.453592,
            ),  # "50 lb total"
            (r"total\s+(?:rocket\s+)?(?:weight|mass|weighs?)\s+(\d+\.?\d*)", 0.453592),
            (r"rocket\s+weighs?\s+(\d+\.?\d*)", 0.453592),
            (r"(\d+\.?\d*)\s*kg\s+total", 1.0),
            (r"(\d+\.?\d*)\s*lb\s+total", 0.453592),
        ]
        for pattern, multiplier in total_mass_patterns:
            match = re.search(pattern, text_normalized)
            if match:
                value = float(match.group(1))
                req.total_rocket_mass_kg = value * multiplier
                break

        # Extract payload mass (only if explicitly mentioned as payload)
        payload_mass_patterns = [
            (r"payload\s+(?:mass|weight|weighs?)\s+(\d+\.?\d*)", 0.453592),
            (r"(\d+\.?\d*)\s*(?:lb|lbs|pound|pounds)\s+payload", 0.453592),
            (r"(\d+\.?\d*)\s*kg\s+payload", 1.0),
            (r"carries?\s+(\d+\.?\d*)\s*(?:lb|lbs|pound|pounds)", 0.453592),  # "carries 10 lbs"
            (r"carries?\s+(\d+\.?\d*)\s*kg", 1.0),  # "carries 10 kg"
        ]
        for pattern, multiplier in payload_mass_patterns:
            match = re.search(pattern, text_normalized)
            if match:
                value = float(match.group(1))
                if "kg" in pattern or "kilogram" in pattern:
                    req.payload_mass_kg = value
                else:
                    req.payload_mass_kg = value * multiplier
                break

        # Extract payload size (cubesat format)
        payload_patterns = [
            r"(\d+[Uu])\b",  # 6U, 3U, etc.
            r"(\d+)\s*unit",
            r"cubesat",
        ]
        for pattern in payload_patterns:
            match = re.search(pattern, text_normalized)
            if match:
                if "U" in match.group(1) or "u" in match.group(1):
                    req.payload_size = match.group(1).upper()
                elif "cubesat" in text_normalized:
                    req.payload_size = "1U"
                break

        # Extract diameter constraint - handle inches, cm, mm
        diameter_patterns = [
            (r"(\d+\.?\d*)\s*inch", 0.0254),
            (r"(\d+\.?\d*)\s*in\b(?!\w)", 0.0254),
            (r"(\d+\.?\d*)\s*cm\b", 0.01),
            (r"(\d+\.?\d*)\s*centimeters?", 0.01),
            (r"(\d+\.?\d*)\s*mm\b", 0.001),
            (r"(\d+\.?\d*)\s*millimeters?", 0.001),
            (r"diameter\s+of\s+(\d+\.?\d*)", 0.0254),  # Assume inches
            (r"max\s+diameter\s+(\d+\.?\d*)", 0.0254),
            (r"(\d+\.?\d*)\s*inch\s+diameter", 0.0254),
        ]
        for pattern, multiplier in diameter_patterns:
            match = re.search(pattern, text_normalized)
            if match:
                value = float(match.group(1))
                req.diameter_constraint_m = value * multiplier
                break

        # Extract length constraint
        length_patterns = [
            (r"length\s+of\s+(\d+\.?\d*)", 0.3048),  # Assume feet
            (r"max\s+length\s+(\d+\.?\d*)", 0.3048),
            (r"(\d+\.?\d*)\s*ft\s+long", 0.3048),
            (r"(\d+\.?\d*)\s*m\s+long", 1.0),
        ]
        for pattern, multiplier in length_patterns:
            match = re.search(pattern, text_normalized)
            if match:
                value = float(match.group(1))
                req.length_constraint_m = value * multiplier
                break

        # Extract fin count
        fin_count_patterns = [
            r"(\d+)\s+fin",
            r"(\d+)\s+fins",
            r"fin\s+count\s+of\s+(\d+)",
        ]
        for pattern in fin_count_patterns:
            match = re.search(pattern, text_normalized)
            if match:
                req.fin_count = int(match.group(1))
                break

        # Extract recovery method - handle many variations
        if any(term in text_normalized for term in ["dual", "two stage", "two-stage", "drogue"]):
            req.recovery_method = "dual_deploy"
        elif "parachute" in text_normalized or "chute" in text_normalized:
            req.recovery_method = "parachute"
        elif "streamer" in text_normalized:
            req.recovery_method = "streamer"
        elif "no recovery" in text_normalized or "no parachute" in text_normalized:
            req.recovery_method = "none"

        # Extract landing distance
        landing_patterns = [
            (r"land(?:s|ing)?\s+(?:within|less than|under|max)\s+(\d+\.?\d*)\s*mile", 1609.34),
            (r"(\d+\.?\d*)\s*mile\s+(?:from|away)", 1609.34),
            (r"(\d+\.?\d*)\s*km\s+(?:from|away)", 1000.0),
            (r"(\d+\.?\d*)\s*meter\s+(?:from|away)", 1.0),
        ]
        for pattern, multiplier in landing_patterns:
            match = re.search(pattern, text_normalized)
            if match:
                value = float(match.group(1))
                if "mile" in pattern:
                    req.landing_distance_miles = value
                    req.landing_distance_m = value * multiplier
                elif "km" in pattern:
                    req.landing_distance_km = value
                    req.landing_distance_m = value * multiplier
                else:
                    req.landing_distance_m = value
                break

        # Extract max velocity/speed
        velocity_patterns = [
            (r"max\s+(?:velocity|speed)\s+of\s+(\d+\.?\d*)\s*mph", 0.44704),
            (r"max\s+(?:velocity|speed)\s+of\s+(\d+\.?\d*)\s*km/h", 0.277778),
            (r"max\s+(?:velocity|speed)\s+of\s+(\d+\.?\d*)\s*m/s", 1.0),
            (r"(\d+\.?\d*)\s*mph\s+max", 0.44704),
        ]
        for pattern, multiplier in velocity_patterns:
            match = re.search(pattern, text_normalized)
            if match:
                value = float(match.group(1))
                req.max_velocity_ms = value * multiplier
                break

        # Extract motor preference
        motor_patterns = [
            r"([A-O])\d+",  # Motor designation like M1670
            r"class\s+([A-O])",  # Motor class
            r"([A-O])\s+class",
        ]
        for pattern in motor_patterns:
            match = re.search(pattern, text_normalized, re.IGNORECASE)
            if match:
                req.motor_preference = match.group(1).upper()
                break

        # Extract material preferences
        materials = ["fiberglass", "carbon", "cardboard", "aluminum", "plywood", "balsa"]
        for material in materials:
            if material in text_normalized:
                if "body" in text_normalized or "tube" in text_normalized:
                    req.body_material = material.capitalize()
                elif "fin" in text_normalized:
                    req.fin_material = material.capitalize()
                elif "nose" in text_normalized:
                    req.nose_material = material.capitalize()

        return req

    def design_rocket(self, requirements: RocketRequirements) -> Dict:
        """
        Design a comprehensive rocket configuration based on requirements.

        Uses heuristics and physics to determine:
        - Rocket dimensions (constrained by requirements)
        - Motor selection (from database or default)
        - Parachute sizing (based on mass and landing distance)
        - Fin configuration (count, size, shape)
        - Material selection
        - Recovery system design
        """
        config = {}

        # Determine payload dimensions
        if requirements.payload_size and requirements.payload_size in self.CUBESAT_SIZES:
            payload_dims = self.CUBESAT_SIZES[requirements.payload_size]
            config["payload_length"] = payload_dims[2]  # Longest dimension
            config["payload_width"] = payload_dims[1]
            config["payload_height"] = payload_dims[0]
        elif requirements.payload_dimensions:
            config["payload_length"] = requirements.payload_dimensions[0]
            config["payload_width"] = requirements.payload_dimensions[1]
            config["payload_height"] = requirements.payload_dimensions[2]
        else:
            # Default payload bay
            config["payload_length"] = 0.3
            config["payload_width"] = 0.1
            config["payload_height"] = 0.1

        # Estimate required motor impulse first
        rough_diameter = 0.075  # Start with 75mm
        if requirements.diameter_constraint_m:
            rough_diameter = requirements.diameter_constraint_m
        elif config.get("payload_width") and config.get("payload_height"):
            rough_diameter = max(config["payload_width"], config["payload_height"]) * 1.2

        # Iterative design to meet target altitude
        # Start with initial estimate and refine
        target_altitude = requirements.target_altitude_m
        max_iterations = 10
        tolerance = 0.05  # 5% altitude tolerance

        motor = None
        best_config = None
        best_altitude_error = float("inf")
        simulated_altitude = 0.0

        if target_altitude:
            # Iterative design loop
            for iteration in range(max_iterations):
                # Estimate required impulse based on current design
                if iteration == 0:
                    # Initial estimate
                    required_impulse = self._estimate_required_impulse(
                        target_altitude, requirements.payload_mass_kg or 0.0, rough_diameter
                    )
                else:
                    # Refine based on previous iteration's error
                    if simulated_altitude > 0:
                        altitude_error_ratio = (
                            simulated_altitude - target_altitude
                        ) / target_altitude
                        # If too high, reduce impulse; if too low, increase
                        impulse_adjustment = 1.0 - altitude_error_ratio * 0.3  # Damping factor
                        required_impulse *= max(
                            0.5, min(2.0, impulse_adjustment)
                        )  # Limit adjustment

                # Select motor
                motor = self._select_motor(
                    required_impulse,
                    requirements.motor_preference,
                    requirements.motor_designation,
                    requirements.motor_manufacturer,
                )

                if motor is None:
                    break

                # Build rocket with this motor
                config["motor"] = motor

                # Determine rocket diameter - MUST account for total mass requirement
                # For heavy rockets, need larger diameter
                if requirements.diameter_constraint_m:
                    body_radius = requirements.diameter_constraint_m / 2.0
                elif requirements.total_rocket_mass_kg:
                    # Size based on total mass - heavier rockets need larger diameter
                    # Estimate: ~0.01m radius per 5kg of total mass (minimum)
                    estimated_radius = (requirements.total_rocket_mass_kg / 5.0) * 0.01
                    estimated_radius = max(estimated_radius, 0.05)  # Minimum 100mm diameter
                    # Round to standard sizes
                    standard_radii = [
                        0.019,
                        0.027,
                        0.0375,
                        0.049,
                        0.0635,
                        0.076,
                        0.10,
                        0.127,
                        0.15,
                        0.20,
                    ]
                    body_radius = min(standard_radii, key=lambda x: abs(x - estimated_radius))
                    # Make sure it fits the motor
                    if motor and motor.get("diameter"):
                        motor_diameter = motor["diameter"]
                        body_radius = max(body_radius, (motor_diameter / 2.0) + 0.003)
                elif motor and motor.get("diameter"):
                    motor_diameter = motor["diameter"]
                    body_radius = (motor_diameter / 2.0) + 0.003
                    standard_radii = [0.019, 0.027, 0.0375, 0.049, 0.0635, 0.076, 0.10, 0.127]
                    body_radius = min(standard_radii, key=lambda x: abs(x - body_radius))
                    if body_radius < (motor_diameter / 2.0):
                        body_radius = (motor_diameter / 2.0) + 0.003
                else:
                    # Estimate from payload
                    payload_size_estimate = max(
                        config.get("payload_width", 0.1), config.get("payload_height", 0.1)
                    )
                    body_radius = payload_size_estimate * 0.6
                    standard_radii = [0.019, 0.027, 0.0375, 0.049, 0.0635, 0.076, 0.10, 0.127]
                    body_radius = min(standard_radii, key=lambda x: abs(x - body_radius))

                config["body_radius"] = body_radius
                config["body_diameter"] = body_radius * 2.0

                # Determine rocket length - MUST account for total mass
                nose_length = body_radius * 2.0  # 2:1 ratio
                payload_bay_length = config.get("payload_length", 0.3) + 0.1
                motor_length = motor.get("length", 0.64) if motor else 0.64

                # Base body length
                body_length = payload_bay_length + motor_length + 0.2

                # If total mass is specified, ensure body is long enough
                if requirements.total_rocket_mass_kg:
                    # Estimate required volume for structure + payload
                    # Rough estimate: need ~0.1m length per 5kg of mass
                    min_length_for_mass = (requirements.total_rocket_mass_kg / 5.0) * 0.1
                    body_length = max(body_length, min_length_for_mass)

                # Minimum reasonable length
                body_length = max(body_length, 0.5)  # At least 50cm

                # Apply length constraint if specified
                if requirements.length_constraint_m:
                    body_length = min(body_length, requirements.length_constraint_m - nose_length)
                if requirements.rocket_length_m:
                    body_length = min(body_length, requirements.rocket_length_m - nose_length)

                config["nose_length"] = nose_length
                config["body_length"] = body_length
                config["total_length"] = nose_length + body_length

                # Design fins
                fin_count = requirements.fin_count or 4
                fin_span = body_radius * 1.5
                fin_root_chord = body_radius * 1.5

                if requirements.fin_size == "small":
                    fin_span *= 0.8
                    fin_root_chord *= 0.8
                elif requirements.fin_size == "large":
                    fin_span *= 1.2
                    fin_root_chord *= 1.2

                fin_tip_chord = fin_root_chord * 0.5
                fin_sweep = fin_root_chord * 0.3

                config["fin_count"] = fin_count
                config["fin_span"] = fin_span
                config["fin_root_chord"] = fin_root_chord
                config["fin_tip_chord"] = fin_tip_chord
                config["fin_sweep"] = fin_sweep
                config["fin_thickness"] = 0.005

                # Calculate mass
                mass_breakdown = self._calculate_comprehensive_mass(config, requirements)
                config["dry_mass"] = mass_breakdown["total_dry_mass"]
                config["mass_breakdown"] = mass_breakdown
                dry_mass = config["dry_mass"]

                # Estimate altitude using rocket equation approximation
                # More accurate: altitude depends on impulse, mass, drag, and atmospheric conditions
                wet_mass = dry_mass + motor.get("total_mass", 0.0)
                drag_factor = 1.0 + (body_radius * 2.0 - 0.075) / 0.1 * 0.1

                # Improved altitude estimation: accounts for drag losses
                # Based on simplified rocket equation with drag
                impulse_per_kg = motor.get("total_impulse", 0) / wet_mass
                # Empirical: each 1000 N·s/kg gives ~150-200m at low altitude, less at high altitude
                base_altitude_per_impulse = 0.15  # meters per N·s/kg
                estimated_altitude = impulse_per_kg * base_altitude_per_impulse / drag_factor

                # Check convergence
                altitude_error = abs(estimated_altitude - target_altitude) / target_altitude
                simulated_altitude = estimated_altitude

                if altitude_error < tolerance:
                    # Good enough! Ensure mass is set
                    if "dry_mass" not in config or config.get("dry_mass", 0) == 0:
                        mass_breakdown = self._calculate_comprehensive_mass(config, requirements)
                        config["dry_mass"] = mass_breakdown["total_dry_mass"]
                        config["mass_breakdown"] = mass_breakdown
                    break

                # Track best so far
                if altitude_error < best_altitude_error:
                    best_altitude_error = altitude_error
                    best_config = config.copy()
                    best_config["motor"] = motor
                elif iteration > 2 and best_config:
                    # Getting worse after a few iterations, use best config
                    config = best_config.copy()
                    motor = config.get("motor")
                    # Recalculate mass for final config
                    mass_breakdown = self._calculate_comprehensive_mass(config, requirements)
                    config["dry_mass"] = mass_breakdown["total_dry_mass"]
                    config["mass_breakdown"] = mass_breakdown
                    break
        else:
            # No target altitude - use simple design
            required_impulse = 5000.0  # Default M-class
            motor = self._select_motor(
                required_impulse,
                requirements.motor_preference,
                requirements.motor_designation,
                requirements.motor_manufacturer,
            )

        # Fallback to default motor if needed
        if motor is None:
            motor = {
                "designation": "M1670",
                "manufacturer": "Cesaroni",
                "total_impulse": 6026.0,
                "max_thrust": 2200.0,
                "avg_thrust": 1545.0,
                "burn_time": 3.9,
                "diameter": 0.075,
                "length": 0.64,
                "total_mass": 4.771,
                "propellant_mass": 2.956,
                "case_mass": 1.815,
            }

        config["motor"] = motor

        # If we didn't do iterative design, do the design now
        # Also need to ensure mass is calculated even if iterative design ran
        needs_design = not target_altitude or "body_radius" not in config
        needs_mass_calc = "dry_mass" not in config or config.get("dry_mass", 0) == 0

        if needs_design:
            # Determine rocket diameter - MUST account for total mass requirement
            if requirements.diameter_constraint_m:
                body_radius = requirements.diameter_constraint_m / 2.0
            elif requirements.total_rocket_mass_kg:
                # Size based on total mass - heavier rockets need larger diameter
                # Estimate: ~0.01m radius per 5kg of total mass (minimum)
                estimated_radius = (requirements.total_rocket_mass_kg / 5.0) * 0.01
                estimated_radius = max(estimated_radius, 0.05)  # Minimum 100mm diameter
                # Round to standard sizes
                standard_radii = [
                    0.019,
                    0.027,
                    0.0375,
                    0.049,
                    0.0635,
                    0.076,
                    0.10,
                    0.127,
                    0.15,
                    0.20,
                ]
                body_radius = min(standard_radii, key=lambda x: abs(x - estimated_radius))
                # Make sure it fits the motor
                if motor and motor.get("diameter"):
                    motor_diameter = motor["diameter"]
                    body_radius = max(body_radius, (motor_diameter / 2.0) + 0.003)
            elif motor and motor.get("diameter"):
                motor_diameter = motor["diameter"]
                body_radius = (motor_diameter / 2.0) + 0.003
                standard_radii = [0.019, 0.027, 0.0375, 0.049, 0.0635, 0.076, 0.10, 0.127]
                body_radius = min(standard_radii, key=lambda x: abs(x - body_radius))
                if body_radius < (motor_diameter / 2.0):
                    body_radius = (motor_diameter / 2.0) + 0.003
            else:
                # Estimate from payload
                payload_size_estimate = max(
                    config.get("payload_width", 0.1), config.get("payload_height", 0.1)
                )
                body_radius = payload_size_estimate * 0.6
                standard_radii = [0.019, 0.027, 0.0375, 0.049, 0.0635, 0.076, 0.10, 0.127]
                body_radius = min(standard_radii, key=lambda x: abs(x - body_radius))

            config["body_radius"] = body_radius
            config["body_diameter"] = body_radius * 2.0

            # Determine rocket length - MUST account for total mass
            nose_length = body_radius * 2.0  # 2:1 ratio
            payload_bay_length = config.get("payload_length", 0.3) + 0.1
            motor_length = motor.get("length", 0.64) if motor else 0.64

            # Base body length
            body_length = payload_bay_length + motor_length + 0.2

            # If total mass is specified, ensure body is long enough
            if requirements.total_rocket_mass_kg:
                # Estimate required volume for structure + payload
                # Rough estimate: need ~0.1m length per 5kg of mass
                min_length_for_mass = (requirements.total_rocket_mass_kg / 5.0) * 0.1
                body_length = max(body_length, min_length_for_mass)

            # Minimum reasonable length
            body_length = max(body_length, 0.5)  # At least 50cm

            # Apply length constraint if specified
            if requirements.length_constraint_m:
                body_length = min(body_length, requirements.length_constraint_m - nose_length)
            if requirements.rocket_length_m:
                body_length = min(body_length, requirements.rocket_length_m - nose_length)

            config["nose_length"] = nose_length
            config["body_length"] = body_length
            config["total_length"] = nose_length + body_length

            # Design fins - use requirements if specified
            fin_count = requirements.fin_count or 4
            fin_span = body_radius * 1.5
            fin_root_chord = body_radius * 1.5

            # Adjust fin size based on requirement
            if requirements.fin_size == "small":
                fin_span *= 0.8
                fin_root_chord *= 0.8
            elif requirements.fin_size == "large":
                fin_span *= 1.2
                fin_root_chord *= 1.2

            fin_tip_chord = fin_root_chord * 0.5
            fin_sweep = fin_root_chord * 0.3

            config["fin_count"] = fin_count
            config["fin_span"] = fin_span
            config["fin_root_chord"] = fin_root_chord
            config["fin_tip_chord"] = fin_tip_chord
            config["fin_sweep"] = fin_sweep
            config["fin_thickness"] = 0.005

            # Calculate comprehensive mass breakdown
            mass_breakdown = self._calculate_comprehensive_mass(config, requirements)
            config["dry_mass"] = mass_breakdown["total_dry_mass"]
            config["mass_breakdown"] = mass_breakdown
            dry_mass = config["dry_mass"]  # For use in parachute sizing

        # Always recalculate mass if it's missing or zero (safety check)
        if needs_mass_calc or config.get("dry_mass", 0) == 0:
            mass_breakdown = self._calculate_comprehensive_mass(config, requirements)
            config["dry_mass"] = mass_breakdown["total_dry_mass"]
            config["mass_breakdown"] = mass_breakdown
            dry_mass = config.get("dry_mass", 0)
        else:
            dry_mass = config.get("dry_mass", 0)

        # Check if total mass constraint is satisfied
        if requirements.total_rocket_mass_kg:
            wet_mass = dry_mass + (motor.get("total_mass", 0.0) if motor else 0.0)
            if wet_mass > requirements.total_rocket_mass_kg * 1.1:  # 10% tolerance
                # Need to reduce mass - could adjust materials, sizes, etc.
                pass  # TODO: Implement mass reduction logic

        # Design parachutes based on recovery method and landing distance
        recovery_method = requirements.recovery_method or "parachute"

        if recovery_method == "dual_deploy":
            # Drogue for high-speed descent
            drogue_area = self._size_drogue_parachute(
                dry_mass + (motor.get("case_mass", 0.0) if motor else 0.0),
                requirements.target_altitude_m or 3000.0,
            )
            config["drogue_diameter"] = math.sqrt(drogue_area / math.pi) * 2.0
            config["drogue_cd"] = 1.3

            # Main for low-speed descent - size based on landing distance if specified
            if requirements.landing_distance_m:
                # Size main chute to achieve desired landing distance
                main_area = self._size_main_parachute_for_distance(
                    dry_mass + (motor.get("case_mass", 0.0) if motor else 0.0),
                    requirements.landing_distance_m,
                    requirements.target_altitude_m or 3000.0,
                )
            else:
                main_area = self._size_main_parachute(
                    dry_mass + (motor.get("case_mass", 0.0) if motor else 0.0)
                )

            config["main_chute_diameter"] = math.sqrt(main_area / math.pi) * 2.0
            config["main_chute_cd"] = 1.5
            config["has_drogue"] = True
            config["has_main_chute"] = True
        elif recovery_method == "parachute":
            # Single parachute
            if requirements.landing_distance_m:
                main_area = self._size_main_parachute_for_distance(
                    dry_mass + (motor.get("case_mass", 0.0) if motor else 0.0),
                    requirements.landing_distance_m,
                    requirements.target_altitude_m or 3000.0,
                )
            else:
                main_area = self._size_main_parachute(
                    dry_mass + (motor.get("case_mass", 0.0) if motor else 0.0)
                )
            config["main_chute_diameter"] = math.sqrt(main_area / math.pi) * 2.0
            config["main_chute_cd"] = 1.5
            config["has_drogue"] = False
            config["has_main_chute"] = True
        else:
            config["has_drogue"] = False
            config["has_main_chute"] = False

        # Use specified chute sizes if provided
        if requirements.drogue_diameter_m:
            config["drogue_diameter"] = requirements.drogue_diameter_m
        if requirements.main_chute_diameter_m:
            config["main_chute_diameter"] = requirements.main_chute_diameter_m

        # Deployment settings
        if requirements.deployment_altitude_m:
            config["main_deployment_altitude"] = requirements.deployment_altitude_m
        elif requirements.target_altitude_m:
            config["main_deployment_altitude"] = requirements.target_altitude_m * 0.25
        else:
            config["main_deployment_altitude"] = 800.0

        config["main_deployment_event"] = "ALTITUDE"
        config["main_deployment_delay"] = 1.5
        config["drogue_deployment_event"] = "APOGEE"
        config["drogue_deployment_delay"] = 1.5

        # Material selection
        config["nose_material"] = requirements.nose_material or "Fiberglass"
        config["body_material"] = requirements.body_material or "Fiberglass"
        config["fin_material"] = requirements.fin_material or "Fiberglass"
        config["motor_mount_material"] = "Fiberglass"

        # Nosecone shape (default to VON_KARMAN for efficiency)
        config["nose_shape"] = "VON_KARMAN"  # Can be expanded later based on requirements

        # Thickness based on size
        if body_radius > 0.05:
            thickness = 0.003
        else:
            thickness = 0.002

        config["nose_thickness"] = thickness
        config["body_thickness"] = thickness
        config["motor_mount_thickness"] = thickness

        return config

    def _estimate_required_impulse(
        self, target_altitude_m: float, payload_mass_kg: float, diameter_m: float
    ) -> float:
        """Estimate required motor impulse to reach target altitude"""
        # More accurate estimation based on rocket equation and drag
        # Base impulse per 100m varies with altitude (higher = more drag, less efficient)
        # These values are calibrated to avoid oversizing motors
        if target_altitude_m < 1000:
            base_impulse_per_100m = 120.0  # Lower altitude = more efficient
        elif target_altitude_m < 2000:
            base_impulse_per_100m = 150.0
        elif target_altitude_m < 3000:
            base_impulse_per_100m = 180.0
        elif target_altitude_m < 4000:
            base_impulse_per_100m = 220.0
        else:
            base_impulse_per_100m = 260.0  # High altitude = less efficient

        # Scale with payload mass (heavier = more impulse needed, but less than linear)
        mass_factor = 1.0 + (payload_mass_kg / 10.0) * 0.15  # +1.5% per kg

        # Scale with diameter (larger = more drag, but effect is moderate)
        drag_factor = 1.0 + (diameter_m - 0.075) / 0.1 * 0.08  # +0.8% per 10mm

        # Calculate required impulse
        required_impulse = (
            (target_altitude_m / 100.0) * base_impulse_per_100m * mass_factor * drag_factor
        )

        # Don't add excessive margin - just round to nearest reasonable value
        # This prevents oversizing which causes rockets to go too high
        return required_impulse  # Return exact value, let motor selection find closest match

    def _select_motor(
        self,
        required_impulse: float,
        preference: Optional[str] = None,
        designation: Optional[str] = None,
        manufacturer: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Select appropriate motor from database with multiple filters"""
        if not self.motor_database:
            return None

        candidates = self.motor_database

        # Filter by designation if specified
        if designation:
            candidates = [m for m in candidates if designation.upper() in m.designation.upper()]

        # Filter by manufacturer if specified
        if manufacturer:
            candidates = [m for m in candidates if manufacturer.lower() in m.manufacturer.lower()]

        # Filter by preference (motor class) if specified
        if preference:
            candidates = [m for m in candidates if m.designation.startswith(preference.upper())]

        # Find motor with impulse closest to required
        # Prefer motors that are close but not too much over (within 30% over is ideal)
        best_motor = None
        best_score = float("inf")

        for motor in candidates:
            if motor.total_impulse > 0:  # Only consider motors with valid impulse
                # Score based on how close to required, with penalty for being too much over
                if motor.total_impulse >= required_impulse * 0.8:  # At least 80% of required
                    diff = abs(motor.total_impulse - required_impulse)
                    # Prefer motors that are close to required, or slightly over (within 30%)
                    if motor.total_impulse <= required_impulse * 1.3:
                        # Within ideal range - score by absolute difference
                        score = diff
                    else:
                        # Too much over - add penalty
                        score = diff + (motor.total_impulse - required_impulse * 1.3) * 2

                    if score < best_score:
                        best_score = score
                        best_motor = motor

        if best_motor:
            return {
                "designation": best_motor.designation,
                "manufacturer": best_motor.manufacturer,
                "total_impulse": best_motor.total_impulse,
                "max_thrust": best_motor.max_thrust,
                "avg_thrust": best_motor.avg_thrust,
                "burn_time": best_motor.burn_time,
                "diameter": best_motor.diameter,
                "length": best_motor.length,
                "total_mass": best_motor.total_mass,
                "propellant_mass": best_motor.propellant_mass,
                "case_mass": best_motor.case_mass,
                "thrust_curve": best_motor.thrust_curve,
            }

        return None

    def _calculate_comprehensive_mass(self, config: Dict, requirements: RocketRequirements) -> Dict:
        """
        Calculate accurate mass for ALL components including:
        - Nose cone (with accurate shape-based volume)
        - Body tube (with wall thickness)
        - Fins (all fins with accurate geometry)
        - Motor mount tube
        - Centering rings
        - Bulkheads
        - Recovery hardware (parachutes, shock cords, swivels, etc.)
        - Avionics bay (if present)
        - Payload bay
        - Fasteners (screws, rivets, etc.)
        - Adhesives/epoxy
        - Paint/finish
        - Wire harnesses
        - Altimeter/electronics
        """
        body_radius = config["body_radius"]
        body_length = config["body_length"]
        nose_length = config["nose_length"]
        body_thickness = config.get("body_thickness", 0.003)
        nose_thickness = config.get("nose_thickness", 0.003)

        # Material densities
        body_density = self.MATERIAL_DENSITIES.get(
            config.get("body_material", "Fiberglass"), 1850.0
        )
        nose_density = self.MATERIAL_DENSITIES.get(
            config.get("nose_material", "Fiberglass"), 1850.0
        )
        fin_density = self.MATERIAL_DENSITIES.get(config.get("fin_material", "Fiberglass"), 1850.0)
        mount_density = self.MATERIAL_DENSITIES.get(
            config.get("motor_mount_material", "Fiberglass"), 1850.0
        )

        mass_breakdown = {}

        # 1. Nose cone - accurate shape-based calculation
        nose_shape = config.get("nose_shape", "VON_KARMAN")
        if nose_shape == "CONICAL":
            nose_outer_vol = (1 / 3) * math.pi * body_radius**2 * nose_length
        elif nose_shape == "OGIVE":
            # Tangent ogive approximation
            nose_outer_vol = 0.5 * math.pi * body_radius**2 * nose_length
        elif nose_shape == "VON_KARMAN":
            # Von Karman ogive - more accurate
            nose_outer_vol = 0.5 * math.pi * body_radius**2 * nose_length * 0.95
        else:
            # Generic approximation
            nose_outer_vol = 0.5 * math.pi * body_radius**2 * nose_length

        # Subtract hollow interior
        if nose_thickness > 0:
            inner_radius = body_radius - nose_thickness
            scale = (inner_radius / body_radius) ** 2
            nose_inner_vol = nose_outer_vol * scale
            nose_vol = nose_outer_vol - nose_inner_vol
        else:
            nose_vol = nose_outer_vol

        mass_breakdown["nose_cone"] = nose_vol * nose_density

        # 2. Body tube - accurate tube calculation
        inner_radius = body_radius - body_thickness
        body_vol = math.pi * body_length * (body_radius**2 - inner_radius**2)
        mass_breakdown["body_tube"] = body_vol * body_density

        # 3. Fins - accurate trapezoidal calculation
        fin_count = config.get("fin_count", 4)
        fin_root = config.get("fin_root_chord", 0.12)
        fin_tip = config.get("fin_tip_chord", 0.06)
        fin_span = config.get("fin_span", 0.11)
        fin_thickness = config.get("fin_thickness", 0.005)

        fin_area = 0.5 * (fin_root + fin_tip) * fin_span
        single_fin_vol = fin_area * fin_thickness
        mass_breakdown["fins"] = single_fin_vol * fin_density * fin_count

        # 4. Motor mount tube
        motor = config.get("motor", {})
        motor_diameter = motor.get("diameter", 0.075)
        motor_length = motor.get("length", 0.64)
        mount_radius = (motor_diameter / 2.0) + 0.005  # Clearance
        mount_thickness = config.get("motor_mount_thickness", 0.003)
        mount_length = motor_length + 0.1  # Extra length

        mount_inner_radius = mount_radius - mount_thickness
        mount_vol = math.pi * mount_length * (mount_radius**2 - mount_inner_radius**2)
        mass_breakdown["motor_mount_tube"] = mount_vol * mount_density

        # 5. Centering rings (typically 2-3 rings)
        ring_count = 3
        ring_thickness = 0.006  # 6mm thick rings
        ring_vol_per_ring = math.pi * ring_thickness * (body_radius**2 - mount_radius**2)
        mass_breakdown["centering_rings"] = ring_vol_per_ring * mount_density * ring_count

        # 6. Bulkheads (forward and aft)
        bulkhead_thickness = 0.003
        bulkhead_vol = math.pi * bulkhead_thickness * body_radius**2
        mass_breakdown["bulkheads"] = bulkhead_vol * body_density * 2  # Forward and aft

        # 7. Recovery hardware
        # Parachutes (nylon fabric ~50g/m²)
        main_chute_dia = config.get("main_chute_diameter", 2.91)
        main_chute_area = math.pi * (main_chute_dia / 2.0) ** 2
        main_chute_mass = main_chute_area * 0.05  # 50g/m²

        # Packed parachute volume (parachutes pack to ~1/25th diameter, ~2x diameter length - realistic packing)
        main_chute_packed_dia = main_chute_dia / 25.0  # Much smaller - realistic packing ratio
        main_chute_packed_length = main_chute_packed_dia * 2.0  # Packed length is about 2x diameter
        main_chute_packed_vol = (
            math.pi * (main_chute_packed_dia / 2.0) ** 2 * main_chute_packed_length
        )
        mass_breakdown["main_parachute_volume"] = main_chute_packed_vol

        drogue_chute_dia = config.get("drogue_diameter", 0.99)
        drogue_chute_area = math.pi * (drogue_chute_dia / 2.0) ** 2
        drogue_chute_mass = drogue_chute_area * 0.05

        # Packed drogue volume
        drogue_chute_packed_dia = drogue_chute_dia / 25.0  # Much smaller - realistic packing ratio
        drogue_chute_packed_length = drogue_chute_packed_dia * 2.0
        drogue_chute_packed_vol = (
            math.pi * (drogue_chute_packed_dia / 2.0) ** 2 * drogue_chute_packed_length
        )
        mass_breakdown["drogue_parachute_volume"] = (
            drogue_chute_packed_vol if config.get("has_drogue", False) else 0.0
        )

        # Shock cords (nylon webbing ~10g/m)
        shock_cord_length = (nose_length + body_length) * 1.5  # 1.5x rocket length
        shock_cord_mass = shock_cord_length * 0.01  # 10g/m

        # Deployment bags (nylon, ~20g each)
        deployment_bag_mass = 0.02 * (1 + (1 if config.get("has_drogue", False) else 0))

        # Swivels, quick links, etc.
        recovery_hardware_mass = 0.05  # 50g for swivels, quick links, etc.

        mass_breakdown["recovery_system"] = (
            main_chute_mass
            + drogue_chute_mass
            + shock_cord_mass
            + deployment_bag_mass
            + recovery_hardware_mass
        )

        # 8. Avionics bay (if dual deploy)
        if config.get("has_drogue", False):
            avionics_bay_length = 0.15  # 15cm avionics bay
            avionics_bay_vol = (
                math.pi * avionics_bay_length * (body_radius**2 - (body_radius - 0.002) ** 2)
            )
            mass_breakdown["avionics_bay"] = (
                avionics_bay_vol * body_density + 0.1
            )  # +100g for electronics
        else:
            mass_breakdown["avionics_bay"] = 0.0

        # 9. Payload bay
        payload_mass = requirements.payload_mass_kg or 0.0
        payload_bay_length = config.get("payload_length", 0.3)
        payload_bay_vol = (
            math.pi * payload_bay_length * (body_radius**2 - (body_radius - 0.002) ** 2)
        )
        mass_breakdown["payload_bay"] = payload_bay_vol * body_density + payload_mass

        # 10. Fasteners (screws, rivets, etc.)
        # Estimate: ~20 fasteners at ~2g each
        mass_breakdown["fasteners"] = 0.04  # 40g

        # 11. Adhesives/epoxy
        # Estimate: ~5% of structural mass
        structural_mass = (
            mass_breakdown["nose_cone"]
            + mass_breakdown["body_tube"]
            + mass_breakdown["fins"]
            + mass_breakdown["motor_mount_tube"]
            + mass_breakdown["centering_rings"]
            + mass_breakdown["bulkheads"]
        )
        mass_breakdown["adhesives"] = structural_mass * 0.05

        # 12. Paint/finish
        # Estimate: ~100g for primer + paint
        mass_breakdown["paint"] = 0.1

        # 13. Wire harnesses
        mass_breakdown["wiring"] = 0.02  # 20g

        # Total dry mass
        total_dry_mass = sum(mass_breakdown.values())
        mass_breakdown["total_dry_mass"] = total_dry_mass

        return mass_breakdown

    def _estimate_dry_mass(
        self,
        nose_length: float,
        body_length: float,
        body_radius: float,
        fin_span: float,
        fin_root: float,
        fin_tip: float,
        payload_mass: float,
        fin_count: int = 4,
    ) -> float:
        """Legacy method - use _calculate_comprehensive_mass for accurate results"""
        # Quick estimate for backward compatibility
        nose_volume = math.pi * body_radius**2 * nose_length / 3.0
        nose_mass = nose_volume * self.MATERIAL_DENSITIES["Fiberglass"] * 0.1

        body_volume = math.pi * body_radius**2 * body_length
        body_mass = body_volume * self.MATERIAL_DENSITIES["Fiberglass"] * 0.05

        fin_area = 0.5 * (fin_root + fin_tip) * fin_span
        fin_mass = fin_area * 0.005 * self.MATERIAL_DENSITIES["Fiberglass"] * fin_count

        mount_mass = 0.2
        recovery_mass = 0.3

        total = nose_mass + body_mass + fin_mass + mount_mass + recovery_mass + payload_mass
        return total

    def _size_drogue_parachute(self, mass_kg: float, altitude_m: float) -> float:
        """Size drogue parachute for high-speed descent"""
        target_velocity = 40.0  # m/s
        rho = 0.5  # kg/m³ at high altitude
        cd = 1.3
        area = (2.0 * mass_kg * 9.81) / (rho * target_velocity**2 * cd)
        return max(area, 0.5)

    def _size_main_parachute(self, mass_kg: float) -> float:
        """Size main parachute for low-speed descent"""
        target_velocity = 5.0  # m/s
        rho = 1.225  # kg/m³ at sea level
        cd = 1.5
        area = (2.0 * mass_kg * 9.81) / (rho * target_velocity**2 * cd)
        return max(area, 1.0)

    def _size_main_parachute_for_distance(
        self, mass_kg: float, max_distance_m: float, apogee_m: float
    ) -> float:
        """Size main parachute to achieve desired landing distance"""
        # Simplified: larger chute = slower descent = less drift
        # This is a rough heuristic - actual drift depends on wind, altitude, etc.
        base_area = self._size_main_parachute(mass_kg)

        # Estimate: if we want to land within X meters, we need slower descent
        # Rough approximation: drift ~ descent_rate * wind_speed * time
        # For now, scale chute size based on distance requirement
        if max_distance_m < 500:
            # Very tight landing - need larger chute
            return base_area * 1.5
        elif max_distance_m < 1000:
            return base_area * 1.2
        else:
            return base_area

    def build_rocket_config(self, requirements: RocketRequirements) -> Tuple[Dict, Optional[Dict]]:
        """
        Complete pipeline: parse requirements and build rocket configuration
        compatible with the Streamlit app.
        """
        design = self.design_rocket(requirements)

        # Convert to Streamlit app format
        config = {
            "name": "AI-Designed Rocket",
            "has_nose": True,
            "nose_length": design["nose_length"],
            "nose_thickness": design["nose_thickness"],
            "nose_shape": design.get("nose_shape", "VON_KARMAN"),
            "nose_material": design["nose_material"],
            "body_length": design["body_length"],
            "body_radius": design["body_radius"],
            "body_thickness": design["body_thickness"],
            "body_material": design["body_material"],
            "has_fins": True,
            "fin_count": design["fin_count"],
            "fin_root_chord": design["fin_root_chord"],
            "fin_tip_chord": design["fin_tip_chord"],
            "fin_span": design["fin_span"],
            "fin_sweep": design["fin_sweep"],
            "fin_thickness": design["fin_thickness"],
            "fin_material": design["fin_material"],
            "has_motor_mount": True,
            "motor_mount_length": (design.get("motor") or {}).get("length", 0.64) + 0.1,
            "motor_mount_radius": (design.get("motor") or {}).get("diameter", 0.075) / 2.0 + 0.005,
            "motor_mount_thickness": design["motor_mount_thickness"],
            "motor_mount_material": design["motor_mount_material"],
            "has_main_chute": design.get("has_main_chute", True),
            "main_chute_diameter": design.get("main_chute_diameter", 2.91),
            "main_chute_cd": design.get("main_chute_cd", 1.5),
            "main_deployment_event": design.get("main_deployment_event", "ALTITUDE"),
            "main_deployment_altitude": design.get("main_deployment_altitude", 800.0),
            "main_deployment_delay": design.get("main_deployment_delay", 1.5),
            "has_drogue": design.get("has_drogue", False),
            "drogue_diameter": design.get("drogue_diameter", 0.99),
            "drogue_cd": design.get("drogue_cd", 1.3),
            "drogue_deployment_event": design.get("drogue_deployment_event", "APOGEE"),
            "drogue_deployment_altitude": design.get("drogue_deployment_altitude", 0.0),
            "drogue_deployment_delay": design.get("drogue_deployment_delay", 1.5),
        }

        # Generate manufacturing-ready specifications
        config["manufacturing_specs"] = self._generate_manufacturing_specs(
            config, design.get("motor")
        )
        # Get mass breakdown from design if available
        design_mass_breakdown = design.get("mass_breakdown", {})
        config["bom"] = self._generate_bom(config, design.get("motor"), design_mass_breakdown)
        config["space_claims"] = self._generate_space_claims(config)

        return config, design.get("motor")

    def _generate_manufacturing_specs(self, config: Dict, motor: Optional[Dict]) -> Dict:
        """Generate manufacturing specifications: tolerances, clearances, surface finish, etc."""
        body_radius = config["body_radius"]
        body_diameter = body_radius * 2.0

        specs = {
            "tolerances": {
                "diameter": f"±{body_diameter * 0.001 * 1000:.2f} mm",  # ±0.1% of diameter
                "length": "±1.0 mm",
                "wall_thickness": "±0.1 mm",
                "fin_alignment": "±0.5°",
                "fin_attachment": "±0.5 mm",
            },
            "clearances": {
                "motor_to_mount": "0.5 mm radial clearance",
                "centering_ring_fit": "0.1 mm interference fit",
                "nose_cone_to_body": "0.2 mm clearance",
                "parachute_bag_clearance": "10 mm minimum",
            },
            "surface_finish": {
                "body_tube": "Smooth (Ra < 0.8 μm)",
                "nose_cone": "Smooth (Ra < 0.8 μm)",
                "fins": "Smooth (Ra < 1.6 μm)",
            },
            "material_specifications": {
                "body_material": config.get("body_material", "Fiberglass"),
                "nose_material": config.get("nose_material", "Fiberglass"),
                "fin_material": config.get("fin_material", "Fiberglass"),
                "mount_material": config.get("motor_mount_material", "Fiberglass"),
            },
            "assembly_requirements": {
                "adhesive_type": "Epoxy (aerospace grade)",
                "cure_time": "24 hours minimum",
                "bond_line_thickness": "0.1-0.3 mm",
                "fastener_torque": "As per manufacturer spec",
            },
            "quality_requirements": {
                "cg_balance": "Within 1 cal of stability",
                "fin_cant": "< 0.5°",
                "body_straightness": "< 0.5 mm over full length",
                "surface_quality": "No visible defects, smooth finish",
            },
        }

        return specs

    def _generate_bom(
        self, config: Dict, motor: Optional[Dict], mass_breakdown: Dict
    ) -> List[Dict]:
        """Generate comprehensive Bill of Materials"""
        bom = []

        # 1. Nose Cone
        bom.append(
            {
                "part_number": "NC-001",
                "description": f"Nose Cone ({config.get('nose_shape', 'VON_KARMAN')})",
                "quantity": 1,
                "material": config.get("nose_material", "Fiberglass"),
                "dimensions": f"Length: {config['nose_length'] * 1000:.1f} mm, Base Diameter: {config['body_radius'] * 2 * 1000:.1f} mm",
                "thickness": f"{config.get('nose_thickness', 0.003) * 1000:.2f} mm",
                "mass_kg": mass_breakdown.get("nose_cone", 0),
                "notes": f"Shape: {config.get('nose_shape', 'VON_KARMAN')}, Wall thickness: {config.get('nose_thickness', 0.003) * 1000:.2f} mm",
            }
        )

        # 2. Body Tube
        bom.append(
            {
                "part_number": "BT-001",
                "description": "Body Tube",
                "quantity": 1,
                "material": config.get("body_material", "Fiberglass"),
                "dimensions": f"Length: {config['body_length'] * 1000:.1f} mm, OD: {config['body_radius'] * 2 * 1000:.1f} mm",
                "thickness": f"{config.get('body_thickness', 0.003) * 1000:.2f} mm",
                "mass_kg": mass_breakdown.get("body_tube", 0),
                "notes": "Main structural tube",
            }
        )

        # 3. Fins
        fin_count = config.get("fin_count", 4)
        bom.append(
            {
                "part_number": "FIN-001",
                "description": "Trapezoidal Fins",
                "quantity": fin_count,
                "material": config.get("fin_material", "Fiberglass"),
                "dimensions": f"Root: {config.get('fin_root_chord', 0.12) * 1000:.1f} mm, Tip: {config.get('fin_tip_chord', 0.06) * 1000:.1f} mm, Span: {config.get('fin_span', 0.11) * 1000:.1f} mm",
                "thickness": f"{config.get('fin_thickness', 0.005) * 1000:.2f} mm",
                "mass_kg": mass_breakdown.get("fins", 0) / fin_count if fin_count > 0 else 0,
                "notes": f"Set of {fin_count} fins, sweep: {config.get('fin_sweep', 0.06) * 1000:.1f} mm",
            }
        )

        # 4. Motor Mount Tube
        motor_dia = motor.get("diameter", 0.075) if motor else 0.075
        mount_radius = (motor_dia / 2.0) + 0.005
        mount_length = (motor.get("length", 0.64) if motor else 0.64) + 0.1
        bom.append(
            {
                "part_number": "MMT-001",
                "description": "Motor Mount Tube",
                "quantity": 1,
                "material": config.get("motor_mount_material", "Fiberglass"),
                "dimensions": f"Length: {mount_length * 1000:.1f} mm, OD: {mount_radius * 2 * 1000:.1f} mm",
                "thickness": f"{config.get('motor_mount_thickness', 0.003) * 1000:.2f} mm",
                "mass_kg": mass_breakdown.get("motor_mount_tube", 0),
                "notes": f"Fits motor diameter: {motor_dia * 1000:.1f} mm",
            }
        )

        # 5. Centering Rings
        bom.append(
            {
                "part_number": "CR-001",
                "description": "Centering Rings",
                "quantity": 3,
                "material": config.get("motor_mount_material", "Fiberglass"),
                "dimensions": f"OD: {config['body_radius'] * 2 * 1000:.1f} mm, ID: {mount_radius * 2 * 1000:.1f} mm, Thickness: 6.0 mm",
                "thickness": "6.0 mm",
                "mass_kg": mass_breakdown.get("centering_rings", 0) / 3,
                "notes": "3 rings for motor mount support",
            }
        )

        # 6. Bulkheads
        bom.append(
            {
                "part_number": "BH-001",
                "description": "Bulkheads (Forward & Aft)",
                "quantity": 2,
                "material": config.get("body_material", "Fiberglass"),
                "dimensions": f"Diameter: {config['body_radius'] * 2 * 1000:.1f} mm, Thickness: 3.0 mm",
                "thickness": "3.0 mm",
                "mass_kg": mass_breakdown.get("bulkheads", 0) / 2,
                "notes": "Forward and aft bulkheads",
            }
        )

        # 7. Recovery System
        if config.get("has_main_chute", True):
            bom.append(
                {
                    "part_number": "CHUTE-MAIN",
                    "description": "Main Parachute",
                    "quantity": 1,
                    "material": "Nylon Fabric",
                    "dimensions": f"Diameter: {config.get('main_chute_diameter', 2.91) * 1000:.0f} mm",
                    "thickness": "N/A",
                    "mass_kg": 0.0,  # Calculated separately
                    "notes": f"CD: {config.get('main_chute_cd', 1.5)}, Deployment: {config.get('main_deployment_event', 'ALTITUDE')}",
                }
            )

        if config.get("has_drogue", False):
            bom.append(
                {
                    "part_number": "CHUTE-DROGUE",
                    "description": "Drogue Parachute",
                    "quantity": 1,
                    "material": "Nylon Fabric",
                    "dimensions": f"Diameter: {config.get('drogue_diameter', 0.99) * 1000:.0f} mm",
                    "thickness": "N/A",
                    "mass_kg": 0.0,
                    "notes": f"CD: {config.get('drogue_cd', 1.3)}, Deployment: {config.get('drogue_deployment_event', 'APOGEE')}",
                }
            )

        bom.append(
            {
                "part_number": "RECOVERY-HW",
                "description": "Recovery Hardware",
                "quantity": 1,
                "material": "Nylon Webbing, Swivels, Quick Links",
                "dimensions": f"Shock cord length: {(config['nose_length'] + config['body_length']) * 1.5 * 1000:.0f} mm",
                "thickness": "N/A",
                "mass_kg": mass_breakdown.get("recovery_system", 0),
                "notes": "Shock cords, swivels, quick links, deployment bags",
            }
        )

        # 8. Avionics (if dual deploy)
        if config.get("has_drogue", False):
            bom.append(
                {
                    "part_number": "AVIONICS-001",
                    "description": "Avionics Bay",
                    "quantity": 1,
                    "material": config.get("body_material", "Fiberglass"),
                    "dimensions": f"Length: 150 mm, Diameter: {config['body_radius'] * 2 * 1000:.1f} mm",
                    "thickness": "2.0 mm",
                    "mass_kg": mass_breakdown.get("avionics_bay", 0),
                    "notes": "Altimeter, battery, deployment charges",
                }
            )

        # 9. Fasteners
        bom.append(
            {
                "part_number": "FAST-001",
                "description": "Fasteners (Screws, Rivets)",
                "quantity": 20,
                "material": "Stainless Steel",
                "dimensions": "Various sizes",
                "thickness": "N/A",
                "mass_kg": mass_breakdown.get("fasteners", 0),
                "notes": "Screws, rivets, and hardware for assembly",
            }
        )

        # 10. Adhesives
        bom.append(
            {
                "part_number": "ADH-001",
                "description": "Epoxy Adhesive",
                "quantity": 1,
                "material": "Aerospace Grade Epoxy",
                "dimensions": "As required",
                "thickness": "N/A",
                "mass_kg": mass_breakdown.get("adhesives", 0),
                "notes": "For bonding components",
            }
        )

        # 11. Paint/Finish
        bom.append(
            {
                "part_number": "PAINT-001",
                "description": "Paint & Primer",
                "quantity": 1,
                "material": "Aerospace Paint",
                "dimensions": "As required",
                "thickness": "N/A",
                "mass_kg": mass_breakdown.get("paint", 0),
                "notes": "Primer and finish coat",
            }
        )

        # 12. Motor
        if motor:
            bom.append(
                {
                    "part_number": f"MOTOR-{motor.get('designation', 'UNKNOWN')}",
                    "description": f"Motor: {motor.get('designation', 'Unknown')}",
                    "quantity": 1,
                    "material": "Composite Propellant",
                    "dimensions": f"Diameter: {motor.get('diameter', 0) * 1000:.1f} mm, Length: {motor.get('length', 0) * 1000:.1f} mm",
                    "thickness": "N/A",
                    "mass_kg": motor.get("total_mass", 0),
                    "notes": f"Manufacturer: {motor.get('manufacturer', 'Unknown')}, Impulse: {motor.get('total_impulse', 0):.0f} N·s",
                }
            )

        return bom

    def _generate_space_claims(self, config: Dict) -> Dict:
        """Generate comprehensive space claims: component positions, clearances, assembly order"""
        nose_length = config["nose_length"]
        body_length = config["body_length"]
        body_radius = config["body_radius"]
        motor = config.get("motor", {})
        motor_length = motor.get("length", 0.64) if motor else 0.64
        motor_diameter = motor.get("diameter", 0.075) if motor else 0.075

        # Get motor mount dimensions from config or calculate
        motor_mount_radius = config.get("motor_mount_radius", (motor_diameter / 2.0) + 0.005)
        motor_mount_length = config.get("motor_mount_length", motor_length + 0.1)
        motor_start_x = nose_length + body_length - motor_mount_length

        # Payload bay
        payload_length = config.get("payload_length", 0.3)
        payload_start_x = nose_length + 0.1

        # Avionics bay (if dual deploy)
        has_drogue = config.get("has_drogue", False)
        avionics_length = 0.15 if has_drogue else 0.0
        avionics_start_x = nose_length + body_length * 0.3 if has_drogue else None

        # Parachute locations
        main_chute_dia = config.get("main_chute_diameter", 2.91)
        drogue_chute_dia = config.get("drogue_diameter", 0.99)

        space_claims = {
            "component_positions": {
                "nose_cone": {
                    "start": 0.0,
                    "end": nose_length,
                    "diameter": body_radius * 2.0,
                    "length": nose_length,
                    "clearance_required": "0.2 mm to body tube",
                    "notes": f"Shape: {config.get('nose_shape', 'VON_KARMAN')}, Contains main parachute",
                },
                "body_tube": {
                    "start": nose_length,
                    "end": nose_length + body_length,
                    "diameter": body_radius * 2.0,
                    "length": body_length,
                    "clearance_required": "N/A (outermost component)",
                    "notes": "Main structural tube",
                },
                "payload_bay": {
                    "start": payload_start_x,
                    "end": payload_start_x + payload_length,
                    "diameter": body_radius * 2.0 - 0.004,
                    "length": payload_length,
                    "clearance_required": "2 mm wall clearance on all sides",
                    "notes": f"Payload space: {payload_length * 1000:.0f} mm long",
                },
                "avionics_bay": {
                    "start": avionics_start_x if has_drogue else None,
                    "end": avionics_start_x + avionics_length if has_drogue else None,
                    "diameter": body_radius * 2.0 - 0.004,
                    "length": avionics_length if has_drogue else 0.0,
                    "clearance_required": "2 mm wall clearance, 5 mm for wire routing",
                    "notes": "Altimeter, battery, deployment charges (dual deploy only)"
                    if has_drogue
                    else "Not used (single deploy)",
                },
                "motor_mount": {
                    "start": motor_start_x,
                    "end": motor_start_x + motor_mount_length,
                    "diameter": motor_mount_radius * 2.0,
                    "length": motor_mount_length,
                    "clearance_required": "0.5 mm radial to body tube",
                    "notes": f"Fits motor: {motor_diameter * 1000:.1f} mm diameter, {motor_length * 1000:.0f} mm long",
                },
                "motor": {
                    "start": motor_start_x + 0.05,
                    "end": motor_start_x + 0.05 + motor_length,
                    "diameter": motor_diameter,
                    "length": motor_length,
                    "clearance_required": "0.5 mm radial clearance in mount",
                    "notes": f"Motor: {motor.get('designation', 'Unknown') if motor else 'Not specified'}",
                },
                "fins": {
                    "start": nose_length + body_length - config.get("fin_root_chord", 0.12),
                    "end": nose_length + body_length,
                    "span": config.get("fin_span", 0.11),
                    "root_chord": config.get("fin_root_chord", 0.12),
                    "clearance_required": "Fins extend beyond body tube",
                    "notes": f"{config.get('fin_count', 4)} fins, flush with body tube surface",
                },
                "main_parachute": {
                    "location": "Nose cone",
                    "deployed_diameter": main_chute_dia,
                    "packed_diameter": main_chute_dia / 25.0,  # Realistic packing ratio
                    "packed_length": (main_chute_dia / 25.0) * 2.0,
                    "packed_volume": math.pi
                    * ((main_chute_dia / 25.0) / 2.0) ** 2
                    * ((main_chute_dia / 25.0) * 2.0),
                    "start": nose_length * 0.3,
                    "end": nose_length * 0.3 + ((main_chute_dia / 25.0) * 2.0),
                    "deployment_altitude": config.get("main_deployment_altitude", 800.0),
                    "deployment_event": config.get("main_deployment_event", "ALTITUDE"),
                    "clearance_required": f"5 mm minimum clearance around packed chute ({main_chute_dia / 25.0 * 1000:.1f} mm diameter)",
                    "notes": f"Packed in nose cone, deploys at {config.get('main_deployment_altitude', 800.0):.0f} m",
                },
                "drogue_parachute": {
                    "location": "Avionics bay" if has_drogue else "Not used",
                    "deployed_diameter": drogue_chute_dia if has_drogue else 0.0,
                    "packed_diameter": drogue_chute_dia / 25.0
                    if has_drogue
                    else 0.0,  # Realistic packing ratio
                    "packed_length": (drogue_chute_dia / 25.0) * 2.0 if has_drogue else 0.0,
                    "packed_volume": math.pi
                    * ((drogue_chute_dia / 25.0) / 2.0) ** 2
                    * ((drogue_chute_dia / 25.0) * 2.0)
                    if has_drogue
                    else 0.0,
                    "start": avionics_start_x + 0.05 if has_drogue else None,
                    "end": avionics_start_x + 0.05 + ((drogue_chute_dia / 25.0) * 2.0)
                    if has_drogue
                    else None,
                    "deployment_altitude": config.get("drogue_deployment_altitude", 0.0),
                    "deployment_event": config.get("drogue_deployment_event", "APOGEE"),
                    "clearance_required": f"5 mm minimum clearance around packed chute ({drogue_chute_dia / 25.0 * 1000:.1f} mm diameter)"
                    if has_drogue
                    else "N/A",
                    "notes": "Packed in avionics bay, deploys at apogee"
                    if has_drogue
                    else "Not used (single deploy)",
                },
                "centering_rings": {
                    "positions": [
                        motor_start_x,
                        motor_start_x + motor_mount_length * 0.5,
                        motor_start_x + motor_mount_length,
                    ],
                    "outer_diameter": body_radius * 2.0,
                    "inner_diameter": motor_mount_radius * 2.0,
                    "thickness": 0.006,
                    "clearance_required": "0.1 mm interference fit to body tube",
                    "notes": "3 rings supporting motor mount",
                },
            },
            "clearances": {
                "motor_to_mount": "0.5 mm radial clearance minimum",
                "nose_cone_to_body": "0.2 mm clearance for slip fit",
                "centering_ring_to_body": "0.1 mm interference fit",
                "parachute_bag_space": f"5 mm minimum clearance around packed parachutes (main: {main_chute_dia / 25.0 * 1000:.1f}mm packed, drogue: {drogue_chute_dia / 25.0 * 1000:.1f}mm packed)",
                "wire_routing": "5 mm clearance for wire harnesses in avionics bay",
                "payload_bay_clearance": "2 mm wall clearance on all sides",
                "avionics_bay_clearance": "2 mm wall clearance, 5 mm for wire routing",
            },
            "assembly_order": [
                "1. Install motor mount tube with centering rings",
                "2. Install forward bulkhead (separates payload bay)",
                "3. Install avionics bay (if dual deploy) with altimeter and battery",
                "4. Install payload bay components",
                "5. Install aft bulkhead (separates motor compartment)",
                "6. Attach fins to body tube (flush alignment)",
                "7. Install recovery system (parachutes, shock cords, swivels)",
                "8. Route deployment wires through avionics bay",
                "9. Attach nose cone (slip fit)",
                "10. Final assembly check and wiring verification",
                "11. Paint and finish",
            ],
            "fit_requirements": {
                "nose_cone": "Slip fit with 0.2 mm clearance to body tube",
                "motor": "0.5 mm radial clearance in mount (loose fit for easy insertion)",
                "centering_rings": "Interference fit (0.1 mm) to body tube, slip fit to motor mount",
                "fins": "Flush with body tube surface, ±0.5 mm alignment, ±0.5° cant angle",
                "bulkheads": "Interference fit (0.1 mm) to body tube",
                "parachute_bags": "5 mm clearance minimum around packed parachutes",
            },
            "volume_claims": {
                "nose_cone_volume": f"{math.pi * body_radius**2 * nose_length * 0.5:.4f} m³ (approximate)",
                "body_tube_volume": f"{math.pi * body_radius**2 * body_length:.4f} m³",
                "payload_bay_volume": f"{math.pi * (body_radius - 0.002) ** 2 * payload_length:.4f} m³",
                "avionics_bay_volume": f"{math.pi * (body_radius - 0.002) ** 2 * avionics_length:.4f} m³"
                if has_drogue
                else "0.0 m³",
                "motor_mount_volume": f"{math.pi * motor_mount_radius**2 * motor_mount_length:.4f} m³",
                "motor_volume": f"{math.pi * (motor_diameter / 2.0) ** 2 * motor_length:.4f} m³"
                if motor
                else "0.0 m³",
            },
        }

        return space_claims


if __name__ == "__main__":
    # Example usage
    designer = RocketDesigner()

    # Test parsing
    text = "I want a rocket that goes to 10000 ft, carries a 6U payload that weighs 10 lbs, has 4 fins, and lands within 1 mile"
    req = designer.parse_requirements(text)
    print(f"Parsed requirements: {req}")

    # Test design
    config, motor = designer.build_rocket_config(req)
    print("\nGenerated rocket configuration:")
    print(f"  Total length: {config['nose_length'] + config['body_length']:.2f} m")
    print(f"  Body diameter: {config['body_radius'] * 2 * 1000:.1f} mm")
    print(f"  Fin count: {config['fin_count']}")
    print(f"  Dry mass: ~{config.get('dry_mass', 0):.2f} kg")
    if motor:
        print(f"  Motor: {motor['designation']} ({motor['manufacturer']})")
        print(f"  Total impulse: {motor['total_impulse']:.0f} N·s")
