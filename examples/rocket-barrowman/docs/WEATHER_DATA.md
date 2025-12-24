# Weather Data Integration

This document describes how to automatically fetch and use real weather data for rocket simulations based on coordinates and datetime.

## Quick Start

```python
from datetime import datetime
from core import Environment

# Create environment from coordinates and datetime
launch_time = datetime(2024, 6, 15, 12, 0)  # June 15, 2024, noon
env = Environment.from_coordinates(
    latitude=33.0,      # Spaceport America
    longitude=-106.5,
    datetime_obj=launch_time,
    elevation=1400.0,   # meters above sea level
)

# Use the environment in your simulation
props = env.air_properties(altitude=1000.0)
wind = env.wind_velocity(altitude=500.0, time=10.0)
```

## Setup

### ECMWF ERA5 Data (Recommended)

ERA5 provides high-quality historical reanalysis data.

1. **Get CDS API credentials:**
   - Register at https://cds.climate.copernicus.eu/
   - Go to "Your profile" → "API key"
   - Copy your API key and URL

2. **Set environment variables:**
   ```bash
   export CDS_API_URL="https://cds.climate.copernicus.eu/api/v2"
   export CDS_API_KEY="your-api-key-here"
   ```

3. **Install dependencies:**
   ```bash
   pip install cdsapi netCDF4
   ```

### NOAA GFS Data

GFS provides forecast data (typically last 10 days). Direct API access requires additional setup.

## Usage

### Automatic Environment Creation

The simplest way is to use `Environment.from_coordinates()`:

```python
from datetime import datetime
from core import Environment

# Spaceport America launch
env = Environment.from_coordinates(
    latitude=33.0,
    longitude=-106.5,
    datetime_obj=datetime(2024, 6, 15, 12, 0),
    elevation=1400.0,
    use_weather_data=True,  # Fetch ERA5 data
    use_nrlmsise=True,      # Use NRLMSISE-00 for high altitudes
)
```

### Manual Weather Data Fetching

If you want more control:

```python
from datetime import datetime
from core.weather_fetcher import fetch_era5_data, create_environment_from_coordinates

# Fetch weather data
launch_time = datetime(2024, 6, 15, 12, 0)
weather_file = fetch_era5_data(
    latitude=33.0,
    longitude=-106.5,
    datetime_obj=launch_time,
    output_file="launch_weather.nc",
)

# Create environment with the data
env, _ = create_environment_from_coordinates(
    latitude=33.0,
    longitude=-106.5,
    datetime_obj=launch_time,
    elevation=1400.0,
)
```

### What Gets Fetched

When you call `Environment.from_coordinates()`, the system:

1. **Fetches ERA5 data** including:
   - Temperature profiles at multiple pressure levels
   - Wind components (u, v) at multiple altitudes
   - Geopotential height
   - Pressure levels

2. **Creates atmospheric model:**
   - Weather data for low altitudes (0-86 km)
   - NRLMSISE-00 for high altitudes (86-1000 km) if enabled
   - Falls back to ISA if weather data unavailable

3. **Extracts wind profile:**
   - Converts wind data to altitude-dependent profile
   - Creates `DynamicWindModel` with real wind conditions

## Data Caching

Weather data files are automatically cached in `core/weather_cache/` to avoid re-downloading. Files are named:
```
era5_{latitude}_{longitude}_{timestamp}.nc
```

## Fallback Behavior

If weather data cannot be fetched:
- Falls back to ISA atmospheric model
- Uses default wind model (power-law profile)
- Simulation continues with standard conditions

## Examples

### Historical Launch Analysis

```python
from datetime import datetime, timedelta
from core import Environment

# Analyze conditions for a week of potential launch windows
base_date = datetime(2024, 6, 15)
for day in range(7):
    launch_time = base_date + timedelta(days=day)
    env = Environment.from_coordinates(
        latitude=33.0,
        longitude=-106.5,
        datetime_obj=launch_time,
        elevation=1400.0,
    )
    # Check conditions at 10 km altitude
    props = env.air_properties(altitude=10000.0)
    print(f"{launch_time.date()}: Density={props['density']:.4f} kg/m³")
```

### Real-Time Launch Conditions

```python
from datetime import datetime
from core import Environment

# Use current conditions
now = datetime.now()
env = Environment.from_coordinates(
    latitude=33.0,
    longitude=-106.5,
    datetime_obj=now,
    elevation=1400.0,
)
```

## Limitations

- **ERA5 data**: Available for historical dates (1940-present), but requires CDS API access
- **GFS data**: Typically only last 10 days, requires proper API setup
- **Download time**: ERA5 data can take several minutes to download
- **File size**: NetCDF files can be large (10-50 MB)

## Troubleshooting

### "CDS API credentials not found"
Set `CDS_API_URL` and `CDS_API_KEY` environment variables.

### "cdsapi package required"
Install with: `pip install cdsapi`

### "netCDF4 package required"
Install with: `pip install netCDF4`

### Download fails
- Check internet connection
- Verify CDS API credentials
- Check if date is valid (ERA5: 1940-present)
- Try a different date/time

## References

- **ECMWF ERA5**: https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5
- **CDS API**: https://cds.climate.copernicus.eu/api-how-to
- **NOAA GFS**: https://www.ncdc.noaa.gov/data-access/model-data/model-datasets/global-forcast-system-gfs

