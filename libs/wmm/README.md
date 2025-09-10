# WMM (World Magnetic Model)

A Rust wrapper for NOAA's World Magnetic Model, providing accurate Earth magnetic field calculations for aerospace navigation and attitude determination.

## Overview

The World Magnetic Model (WMM) is the standard model used by the U.S. Department of Defense, NATO, and International Hydrographic Organization for navigation, attitude determination, and heading reference systems. This library provides:

- **Magnetic Field Calculations** - Compute magnetic field vectors at any location on Earth
- **Time-Varying Model** - Accounts for secular variation (changes over time)
- **WGS84 Ellipsoid** - Precise Earth shape modeling for accurate calculations
- **Error Estimates** - Provides uncertainty bounds for all field components

## History

The wmm library was developed to support spacecraft attitude determination and control systems (ADCS):

- **[PR #529](https://github.com/elodin-sys/paracosm/pull/529)** (June 2024) - Initial implementation wrapping NOAA's WMM C library as part of adding ADCS simulation capabilities. Integrated with TRIAD algorithm for attitude determination using magnetometer data.

- **[PR #531](https://github.com/elodin-sys/paracosm/pull/531)** (June 2024) - Added integration with `nox-frames` for coordinate transformations between NED, ECEF, and ECI frames. Added `hifitime` support for precise time conversions when calculating reference magnetic field values.

*note*: since this was developed, and better documented and supported [WMM create](https://docs.rs/world_magnetic_model/latest/world_magnetic_model/) is now available. We may migrate to using this in the future.

## Current Usage in Elodin

The wmm library is currently integrated into:

- **MEKF Flight Software** (`fsw/mekf/`) - Provides reference magnetic field vectors for the Multiplicative Extended Kalman Filter on Aleph. Exposed via Lua scripting interface to calculate expected field values at spacecraft position for attitude determination.

- **Related ADCS Components** - Works alongside:
  - `roci/adcs` algorithms (TRIAD, MAG.I.CAL) that compare measured vs. expected fields
  - `fsw/sensor-fw` that reads raw BMM350 magnetometer data
  - Python MEKF implementations in simulation examples

Future integration opportunities include magnetometer calibration routines, orbit determination, and mission planning tools.

## Why WMM is Critical for Flight Software

Spacecraft and drones rely on magnetometers for:
1. **Attitude Determination** - Comparing measured vs. expected magnetic field
2. **Compass Heading** - Calculating magnetic declination for navigation
3. **Magnetometer Calibration** - Removing Earth's field to detect anomalies
4. **Orbit Determination** - Using magnetic field as additional observable

Without accurate magnetic field models, these systems would suffer from:
- Navigation errors up to several degrees
- Poor attitude estimation convergence
- Inability to detect magnetic anomalies
- Degraded sensor fusion performance

## Architecture

```
┌─────────────────────────────────────────┐
│          Your Application               │
│  (ADCS, Navigation, Calibration)        │
└─────────────┬───────────────────────────┘
              │
┌─────────────▼───────────────────────────┐
│          wmm (Rust API)                 │
│  - MagneticModel                        │
│  - GeodeticCoords                       │
│  - Elements (B-field components)        │
└─────────────┬───────────────────────────┘
              │
┌─────────────▼───────────────────────────┐
│     NOAA WMM C Implementation           │
│  - Spherical Harmonic Expansion         │
│  - Coordinate Transformations           │
│  - Secular Variation                    │
└─────────────┬───────────────────────────┘
              │
┌─────────────▼───────────────────────────┐
│      WMM 2020 Coefficients              │
│  - Valid: 2020.0 - 2025.0               │
│  - 12th Order/Degree Model              │
│  - ~3400 nT RMS accuracy                │
└─────────────────────────────────────────┘
```

## Usage

### Basic Field Calculation

```rust
use wmm::{MagneticModel, GeodeticCoords, Epoch};

// Create the magnetic model (WMM 2020)
let mut model = MagneticModel::default();

// Define location: San Francisco
let coords = GeodeticCoords::new(
    37.7749,   // latitude (degrees)
    -122.4194, // longitude (degrees) 
    100.0,     // altitude (meters)
);

// Set the date
let epoch = Epoch::from_gregorian_utc(2024, 6, 1, 0, 0, 0, 0);

// Calculate magnetic field
let (elements, error_bars) = model.calculate_field(epoch, coords);

// Access field components
let [bx, by, bz] = elements.b_field();  // Tesla
println!("B-field: [{:.1}, {:.1}, {:.1}] nT", 
    bx * 1e9, by * 1e9, bz * 1e9);

// Get other useful values
println!("Declination: {:.2}°", elements.declination());
println!("Inclination: {:.2}°", elements.inclination());
println!("Total Intensity: {:.1} nT", elements.total_intensity() * 1e9);
```

### Integration with ADCS

```rust
// Use in attitude determination
fn correct_magnetometer_reading(
    raw_measurement: [f64; 3],
    position: GeodeticCoords,
    time: Epoch,
) -> [f64; 3] {
    let mut model = MagneticModel::default();
    let (elements, _) = model.calculate_field(time, position);
    
    // Subtract Earth's field to get spacecraft field
    let spacecraft_field = [
        raw_measurement[0] - elements.b_field()[0],
        raw_measurement[1] - elements.b_field()[1],
        raw_measurement[2] - elements.b_field()[2],
    ];
    
    spacecraft_field
}
```

## Coordinate Systems

The library uses geodetic coordinates (WGS84):
- **Latitude**: -90° to +90° (positive North)
- **Longitude**: -180° to +180° (positive East)
- **Altitude**: Meters above WGS84 ellipsoid

Output magnetic field components:
- **X**: North component (positive northward)
- **Y**: East component (positive eastward)
- **Z**: Vertical component (positive downward)

## Model Accuracy

| Component | RMS Error | Max Error |
|-----------|-----------|-----------|
| Declination | 0.24° | 1.0° |
| Inclination | 0.21° | 0.8° |
| Horizontal Intensity | 133 nT | 500 nT |
| Total Intensity | 147 nT | 550 nT |

Error increases with:
- Altitude (above 600 km)
- Time from epoch (> 2.5 years)
- Proximity to magnetic poles

## Technical Details

### Spherical Harmonic Model

The WMM uses a 12th degree and order spherical harmonic expansion:

```
V(r,θ,λ,t) = a Σ Σ (a/r)^(n+1) [g_n^m(t)cos(mλ) + h_n^m(t)sin(mλ)] P_n^m(cosθ)
```

Where:
- 90 Gauss coefficients (g, h) define the field
- Secular variation coefficients model time changes
- Associated Legendre functions handle latitude variation

### Build Process

The library uses `bindgen` to wrap the official NOAA C implementation:

```rust
// build.rs
bindgen::Builder::default()
    .header("vendor/GeomagnetismHeader.h")
    .generate()
```

The WMM coefficients are embedded at compile time from `src/coef.rs`.

## Model Updates

The WMM is updated every 5 years by NOAA/NCEI and the British Geological Survey:

- **WMM 2020**: Current model (2020.0 - 2025.0)
- **WMM 2025**: Expected release December 2024
- **Updates**: Typically < 1% change in coefficients

To update coefficients:
1. Download new COF file from NOAA
2. Update coefficients in `src/coef.rs`
3. Adjust epoch in conversion functions
4. Verify against NOAA test values

## Dependencies

- `hifitime` - High-precision time handling
- `approx` - Floating-point comparisons in tests
- `bindgen` (build) - Generate C bindings
- `cc` (build) - Compile C implementation

## References

- [NOAA WMM Page](https://www.ngdc.noaa.gov/geomag/WMM/)
- [WMM Technical Report](https://www.ngdc.noaa.gov/geomag/WMM/data/WMM2020/WMM2020_Report.pdf)
- [Online Calculator](https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml)

## License

The NOAA WMM implementation is in the public domain. See the repository's LICENSE file for the Rust wrapper licensing.
