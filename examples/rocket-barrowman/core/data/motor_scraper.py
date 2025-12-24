"""
API client for thrustcurve.org to extract motor data and thrust curves.
Uses the official ThrustCurve.org API instead of web scraping.
"""

import requests
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time
from dataclasses import dataclass, asdict


@dataclass
class MotorData:
    """Structured motor data from thrustcurve.org API"""

    designation: str
    manufacturer: str
    diameter: float  # meters
    length: float  # meters
    total_mass: float  # kg
    propellant_mass: float  # kg
    case_mass: float  # kg
    burn_time: float  # s
    total_impulse: float  # N·s
    avg_thrust: float  # N
    max_thrust: float  # N
    thrust_curve: List[Tuple[float, float]]  # [(time, thrust), ...]
    delays: List[str]
    url: str


class ThrustCurveScraper:
    """API client for thrustcurve.org motor database"""

    BASE_URL = "https://www.thrustcurve.org"
    API_BASE = f"{BASE_URL}/api/v1"

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path(__file__).parent / "motor_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Elodin-Rocket-Simulator/1.0", "Accept": "application/json"}
        )

    def search_motors(self, query: str = "", max_results: int = 100, **filters) -> List[Dict]:
        """
        Search for motors using ThrustCurve.org API

        Args:
            query: Search query (motor name, designation, etc.) - use 'designation' or 'commonName' in filters instead
            max_results: Maximum number of results to return
            **filters: Additional search filters (availability, impulseClass, diameter, etc.)

        Returns:
            List of motor info dicts with id, designation, manufacturer, etc.
        """
        try:
            # Use the search API endpoint - requires POST with JSON body
            url = f"{self.API_BASE}/search.json"

            # Build request body - API uses POST with JSON, not GET with query params
            payload = {"maxResults": max_results, **filters}

            # Add query to designation or commonName if provided
            if query:
                payload["designation"] = query

            # Remove None values and empty strings
            payload = {k: v for k, v in payload.items() if v is not None and v != ""}

            # POST with JSON body
            response = self.session.post(url, json=payload, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Extract motors from API response
            motors = []
            if "results" in data:
                for motor in data["results"]:
                    motors.append(
                        {
                            "id": motor.get("motorId", ""),
                            "designation": motor.get("designation", ""),
                            "manufacturer": motor.get("manufacturer", {}).get("name", "Unknown")
                            if isinstance(motor.get("manufacturer"), dict)
                            else str(motor.get("manufacturer", "Unknown")),
                            "diameter": motor.get("diameter", 0.0) / 1000.0,  # mm to m
                            "length": motor.get("length", 0.0) / 1000.0,  # mm to m
                            "totalImpulse": motor.get("totalImpulse", 0.0),
                            "avgThrust": motor.get("avgThrust", 0.0),
                            "maxThrust": motor.get("maxThrust", 0.0),
                            "burnTime": motor.get("burnTime", 0.0),
                            "propellantMass": motor.get("propellantMass", 0.0),
                            "totalMass": motor.get("totalMass", 0.0),
                            "caseMass": motor.get("caseMass", 0.0),
                            "availability": motor.get("availability", "unknown"),
                        }
                    )

            return motors

        except Exception as e:
            print(f"Error searching motors: {e}")
            import traceback

            traceback.print_exc()
            return []

    def get_motor_data(
        self, motor_id: str, search_result: Optional[Dict] = None
    ) -> Optional[MotorData]:
        """
        Get detailed motor data using ThrustCurve.org API

        Args:
            motor_id: Motor ID from search results
            search_result: Optional search result dict with motor metadata (designation, manufacturer, etc.)

        Returns:
            MotorData object or None if API call fails
        """
        # Check cache first
        cache_file = self.cache_dir / f"{motor_id}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    # Fix incorrect unit conversions in cached data
                    # Motor diameters are typically 13mm-152mm (0.013-0.152m)
                    # Motor lengths are typically 50mm-2000mm (0.05-2.0m)
                    # If values are suspiciously small, they were likely double-converted
                    diameter = data.get("diameter", 0)
                    length = data.get("length", 0)
                    
                    # If diameter < 0.01m (10mm), it's suspicious - most motors are 13mm+
                    # Likely was incorrectly divided by 1000 when already in meters
                    if 0 < diameter < 0.01:
                        data["diameter"] = diameter * 1000.0
                    # If length < 0.01m (10mm), also suspicious - most motors are 50mm+
                    if 0 < length < 0.01:
                        data["length"] = length * 1000.0
                    
                    return MotorData(**data)
            except Exception:
                pass

        try:
            # Download motor data using API - use POST with JSON body
            url = f"{self.API_BASE}/download.json"

            payload = {
                "motorId": motor_id,
                "data": "samples",  # Get parsed data points
            }

            response = self.session.post(url, json=payload, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Extract thrust curve samples from download response
            # Download API returns 'results' not 'motors'
            download_result = None
            if "results" in data and len(data["results"]) > 0:
                download_result = data["results"][0]
            elif "motors" in data and len(data["motors"]) > 0:
                download_result = data["motors"][0]

            # Get thrust curve samples
            thrust_curve = []
            if download_result and "samples" in download_result:
                for sample in download_result["samples"]:
                    thrust_curve.append(
                        (float(sample.get("time", 0.0)), float(sample.get("thrust", 0.0)))
                    )

            # Calculate total impulse from thrust curve (integral of thrust over time)
            calculated_total_impulse = 0.0
            calculated_avg_thrust = 0.0
            calculated_burn_time = 0.0
            if thrust_curve:
                # Integrate thrust curve to get total impulse
                for i in range(len(thrust_curve) - 1):
                    t1, f1 = thrust_curve[i]
                    t2, f2 = thrust_curve[i + 1]
                    dt = t2 - t1
                    # Trapezoidal rule
                    calculated_total_impulse += (f1 + f2) / 2.0 * dt

                # Burn time is last sample time
                calculated_burn_time = thrust_curve[-1][0] if thrust_curve else 0.0
                # Average thrust = total impulse / burn time
                if calculated_burn_time > 0:
                    calculated_avg_thrust = calculated_total_impulse / calculated_burn_time

            # Use search_result for metadata, fallback to download_result if available
            motor_info = search_result if search_result else download_result
            if not motor_info:
                return None

            # Extract manufacturer name (handle both dict and string formats)
            manufacturer_name = "Unknown"
            if isinstance(motor_info.get("manufacturer"), dict):
                manufacturer_name = motor_info["manufacturer"].get("name", "Unknown")
            elif motor_info.get("manufacturer"):
                manufacturer_name = str(motor_info["manufacturer"])

            # Use calculated values if available, otherwise use from search result
            total_impulse = (
                calculated_total_impulse
                if calculated_total_impulse > 0
                else motor_info.get("totalImpulse", 0.0)
            )
            avg_thrust = (
                calculated_avg_thrust
                if calculated_avg_thrust > 0
                else motor_info.get("avgThrust", 0.0)
            )
            burn_time = (
                calculated_burn_time
                if calculated_burn_time > 0
                else motor_info.get("burnTime", 0.0)
            )
            max_thrust = (
                max([t for _, t in thrust_curve])
                if thrust_curve
                else motor_info.get("maxThrust", 0.0)
            )

            # Extract diameter and length - handle unit conversion correctly
            # If search_result is provided, diameter/length are already in meters (converted in search_motors)
            # If from download_result, they're in mm and need conversion
            if search_result:
                # Already converted to meters in search_motors
                diameter = motor_info.get("diameter", 0.0)
                length = motor_info.get("length", 0.0)
            else:
                # From download API, convert from mm to m
                diameter = motor_info.get("diameter", 0.0) / 1000.0
                length = motor_info.get("length", 0.0) / 1000.0

            # Create MotorData object
            motor_data = MotorData(
                designation=motor_info.get("designation", "Unknown"),
                manufacturer=manufacturer_name,
                diameter=diameter,
                length=length,
                total_mass=motor_info.get("totalMass", 0.0),
                propellant_mass=motor_info.get("propellantMass", 0.0),
                case_mass=motor_info.get("caseMass", 0.0),
                burn_time=burn_time,
                total_impulse=total_impulse,
                avg_thrust=avg_thrust,
                max_thrust=max_thrust,
                thrust_curve=thrust_curve,
                delays=motor_info.get("delays", []),
                url=f"{self.BASE_URL}/motors/{motor_id}.html",
            )

            # Cache the result
            with open(cache_file, "w") as f:
                json.dump(asdict(motor_data), f, indent=2)

            return motor_data

        except Exception as e:
            print(f"Error getting motor data {motor_id}: {e}")
            return None

    def _download_and_parse_data_file(
        self, motor_id: str, format: str = "eng"
    ) -> List[Tuple[float, float]]:
        """Download and parse a data file for a motor"""
        try:
            url = f"{self.API_BASE}/download.json"
            params = {"motorId": motor_id, "format": format}

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()

            # Parse the data file content
            content = response.text
            return self._parse_eng_file(content.split("\n"))

        except Exception as e:
            print(f"Error downloading data file: {e}")
            return []

    def _parse_eng_file(self, lines: List[str]) -> List[Tuple[float, float]]:
        """Parse .eng format (RASP format)"""
        # .eng format: designation diameter length delays [comments]
        #              time thrust [time thrust ...]
        data = []
        in_data = False

        for line in lines:
            line = line.strip()
            if not line or line.startswith(";"):
                continue

            parts = line.split()
            if len(parts) >= 2:
                try:
                    time_val = float(parts[0])
                    thrust_val = float(parts[1])
                    data.append((time_val, thrust_val))
                    in_data = True
                except ValueError:
                    if in_data:
                        break

        return data

    def _parse_rse_file(self, lines: List[str]) -> List[Tuple[float, float]]:
        """Parse .rse format"""
        # Similar to .eng but may have different structure
        return self._parse_eng_file(lines)

    def _parse_csv_data(self, text: str) -> List[Tuple[float, float]]:
        """Parse CSV/TSV data"""
        data = []
        lines = text.strip().split("\n")

        for line in lines[1:]:  # Skip header
            parts = line.split(",") if "," in line else line.split("\t")
            if len(parts) >= 2:
                try:
                    time_val = float(parts[0].strip())
                    thrust_val = float(parts[1].strip())
                    data.append((time_val, thrust_val))
                except ValueError:
                    continue

        return data

    def _parse_generic_data(self, lines: List[str]) -> List[Tuple[float, float]]:
        """Generic data parser"""
        data = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    time_val = float(parts[0])
                    thrust_val = float(parts[1])
                    data.append((time_val, thrust_val))
                except ValueError:
                    continue
        return data

    def scrape_motor_list(
        self, max_motors: int = 10000, impulse_classes: Optional[List[str]] = None
    ) -> List[MotorData]:
        """
        Get a list of motors using the API

        Args:
            max_motors: Maximum number of motors to retrieve (default 10000 to get all)
            impulse_classes: List of impulse classes to filter (e.g., ['M', 'L', 'K'])
                            If None, searches ALL impulse classes

        Returns:
            List of MotorData objects
        """
        motors = []

        # Search for ALL motor classes if not specified
        if impulse_classes is None:
            # All standard impulse classes from A to O
            impulse_classes = [
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "J",
                "K",
                "L",
                "M",
                "N",
                "O",
            ]

        for impulse_class in impulse_classes:
            if len(motors) >= max_motors:
                break

            # Search by impulse class - get up to 500 per class to ensure we get all
            results = self.search_motors(
                query="",
                max_results=min(500, max_motors - len(motors)),
                impulseClass=impulse_class,
                availability="available",  # Only get available motors
            )

            print(f"Found {len(results)} motors for class {impulse_class}")

            for result in results:
                if len(motors) >= max_motors:
                    break

                motor_id = result.get("id")
                if motor_id:
                    # Pass search result to get_motor_data so it has metadata
                    motor_data = self.get_motor_data(motor_id, search_result=result)
                    if motor_data:
                        motors.append(motor_data)
                        if len(motors) % 50 == 0:
                            print(f"Retrieved {len(motors)} motors so far...")

                    # Be polite - rate limiting (API is more forgiving than scraping)
                    time.sleep(0.3)  # Slightly faster since we're getting more motors

        print(f"Total motors retrieved: {len(motors)}")
        return motors

    def save_motor_database(self, motors: List[MotorData], filename: str = "motor_database.json"):
        """Save motor database to JSON file"""
        db_path = self.cache_dir / filename
        data = [asdict(motor) for motor in motors]
        with open(db_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(motors)} motors to {db_path}")

    def load_motor_database(self, filename: str = "motor_database.json") -> List[MotorData]:
        """Load motor database from JSON file"""
        db_path = self.cache_dir / filename
        if not db_path.exists():
            return []

        with open(db_path, "r") as f:
            data = json.load(f)
            return [MotorData(**motor) for motor in data]


if __name__ == "__main__":
    scraper = ThrustCurveScraper()

    # Example: Search for motors
    print("Searching for motors...")
    results = scraper.search_motors("M1670", max_results=10)
    print(f"Found {len(results)} results")

    # Example: Get a specific motor
    if results:
        motor_id = results[0].get("id")
        if motor_id:
            motor = scraper.get_motor_data(motor_id)
            if motor:
                print(f"Retrieved motor: {motor.designation}")
                print(f"  Manufacturer: {motor.manufacturer}")
                print(f"  Total Impulse: {motor.total_impulse:.1f} N·s")
                print(f"  Max Thrust: {motor.max_thrust:.1f} N")
