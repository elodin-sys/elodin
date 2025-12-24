# Atmospheric Models

This document describes the robust atmospheric modeling capabilities available in the rocket-barrowman simulation, similar to RocketPy's environment analysis.

## Available Models

### 1. ISA (International Standard Atmosphere)
**Default model** - Simple, fast, valid up to 86 km.

```python
from core import Environment

env = Environment()
env.set_atmospheric_model(type="ISA")
```

**Use when:**
- Simulating low-altitude flights (< 30 km)
- Quick prototyping
- No specific location/weather data needed

### 2. NRLMSISE-00
**High-altitude model** - Naval Research Laboratory Mass Spectrometer and Incoherent Scatter model. Valid from 0-1000 km.

```python
env.set_atmospheric_model(
    type="NRLMSISE00",
    latitude=35.0,  # Launch site latitude
    longitude=-106.0,  # Launch site longitude
    year=2024,
    day_of_year=150,  # Day of year (1-365)
    f107=150.0,  # Solar flux at 10.7 cm
    f107a=150.0,  # 81-day average of f107
    ap=4.0,  # Geomagnetic activity index
)
```

**Use when:**
- Simulating high-altitude flights (> 50 km)
- Need accurate thermosphere/exosphere properties
- Accounting for solar activity effects

**Installation:**
```bash
pip install nrlmsise00
```

### 3. Weather Data
**Real-world conditions** - Load atmospheric profiles from weather reanalysis or forecast data (ECMWF, NOAA GFS, etc.).

```python
env.set_atmospheric_model(
    type="WEATHER",
    weather_file="path/to/weather_data.nc",  # NetCDF file
)
```

**Use when:**
- Simulating specific launch conditions
- Need realistic weather profiles
- Historical analysis or forecast-based planning

**Data Sources:**
- **ECMWF ERA5**: Historical reanalysis data
- **NOAA GFS**: Global forecast system
- **Weather balloon soundings**: Direct measurements

**Installation:**
```bash
pip install netCDF4
```

### 4. Hybrid Model
**Best of both worlds** - Combines ISA/Weather Data for low altitudes and NRLMSISE-00 for high altitudes.

```python
env.set_atmospheric_model(
    type="HYBRID",
    latitude=35.0,
    longitude=-106.0,
    transition_altitude=86000.0,  # Switch at 86 km
)
```

**Use when:**
- Simulating flights that span multiple atmospheric layers
- Need accuracy at both low and high altitudes
- Best overall accuracy

## Comparison with RocketPy

| Feature | RocketPy | This Implementation |
|---------|----------|-------------------|
| ISA Model | ✅ | ✅ |
| NRLMSISE-00 | ✅ | ✅ |
| Weather Data | ✅ (ECMWF, GFS) | ✅ (NetCDF files) |
| Hybrid Models | ✅ | ✅ |
| Real-time Weather API | ✅ | ⚠️ (via NetCDF files) |
| Sounding Data | ✅ | ⚠️ (via NetCDF files) |

## Example Usage

### Basic ISA Model
```python
from core import Environment

env = Environment(elevation=1400.0)  # Spaceport America elevation
props = env.air_properties(altitude=1000.0)  # 1 km AGL
print(f"Temperature: {props['temperature']:.2f} K")
print(f"Pressure: {props['pressure']:.2f} Pa")
print(f"Density: {props['density']:.4f} kg/m³")
```

### High-Altitude Flight with NRLMSISE-00
```python
env = Environment(elevation=1400.0)
env.set_atmospheric_model(
    type="NRLMSISE00",
    latitude=35.0,
    longitude=-106.0,
    year=2024,
    day_of_year=150,
)

# Get properties at 100 km altitude
props = env.air_properties(altitude=100000.0 - 1400.0)  # 100 km ASL
print(f"Temperature at 100 km: {props['temperature']:.2f} K")
```

### Weather Data Integration
```python
# Download weather data from ECMWF or NOAA
# Save as NetCDF file, then:
env.set_atmospheric_model(
    type="WEATHER",
    weather_file="era5_data.nc",
)
```

## Performance Considerations

- **ISA**: Fastest, suitable for real-time simulations
- **NRLMSISE-00**: Moderate speed, accurate for high altitudes
- **Weather Data**: Slower (file I/O), most realistic
- **Hybrid**: Moderate speed, best accuracy across all altitudes

## Future Enhancements

- Direct API integration with ECMWF/NOAA (no file download needed)
- Automatic weather data fetching for launch sites
- Sounding data integration
- Ensemble forecast support
- Time-varying atmospheric conditions during flight

## References

- **NRLMSISE-00**: [NASA Technical Report](https://ccmc.gsfc.nasa.gov/modelweb/models/nrlmsise00.php)
- **RocketPy Environment**: [Documentation](https://docs.rocketpy.org/en/latest/reference/classes/Environment.html)
- **ECMWF ERA5**: [Data Access](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5)
- **NOAA GFS**: [Data Access](https://www.ncdc.noaa.gov/data-access/model-data/model-datasets/global-forcast-system-gfs)

