# Rocket Simulator Streamlit UI

A web-based interface for creating, simulating, and visualizing rocket flights.

## Features

- **Rocket Builder**: Create custom rockets or use the default Calisto rocket
- **Motor Configuration**: Use default M1670 motor or create custom motors
- **Simulation**: Run 6-DOF flight simulations with configurable parameters
- **Visualization**: Interactive charts and 3D trajectory visualization
- **Elodin Integration**: Launch Elodin editor for advanced 3D visualization
- **Data Export**: Download simulation results as CSV

## Installation

### Prerequisites

**⚠️ IMPORTANT: You MUST run Streamlit from within the nix shell!**

The nix shell provides:
- Correct Python version (3.10+)
- All required dependencies (elodin, jax, etc.)
- Proper environment setup

1. **Enter the nix shell (REQUIRED):**
```bash
cd /home/kush-mahajan/elodin
nix develop
```

You should see your prompt change (may show `[nix-shell:...]` or similar).

**What this does:**
- Provides the correct Python version and Elodin dependencies
- Sets up the development environment
- Required for the simulator to work

4. **Install Streamlit and dependencies:**
```bash
cd examples/rocket-barrowman
./install_deps.sh
```

**What this does:**
- Creates a Python virtual environment (`venv/`)
- Uses `uv` (fast Python package installer) if available, falls back to `pip`
- Installs all required packages from `requirements.txt`

**Note:** The script automatically handles the virtual environment setup. If `uv` is not available, it will use `pip` instead.

5. **Run Streamlit:**
```bash
./run_streamlit.sh
```

**OR run manually:**
```bash
source venv/bin/activate
streamlit run app.py
```

**What to expect:**
- Streamlit will start and show a URL (usually `http://localhost:8501`)
- Your browser should automatically open
- If not, copy the URL from the terminal and paste it in your browser

**⚠️ IMPORTANT:** 
- Make sure you're in the nix shell (`echo $IN_NIX_SHELL` should show a path)
- The `venv/` directory is created by `install_deps.sh` - this is normal and expected
- The virtual environment isolates dependencies from the nix Python environment

## Usage

### Run the Streamlit App

**Recommended (uses helper script):**
```bash
cd examples/rocket-barrowman
./run_streamlit.sh
```

**Or manually:**
```bash
cd examples/rocket-barrowman
source venv/bin/activate
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the UI

1. **Configure Rocket** (Sidebar):
   - Choose "Calisto (Default)" or "Custom Rocket"
   - For custom rockets, configure:
     - Nose cone (length, thickness, material)
     - Body tube (length, radius, thickness, material)
     - Fins (count, dimensions, material)
     - Motor mount
     - Parachutes (main and drogue)

2. **Configure Motor** (Sidebar):
   - Choose "Cesaroni M1670 (Default)" or "Custom Motor"
   - For custom motors, set:
     - Dimensions (diameter, length)
     - Mass properties
     - Thrust curve (max/avg thrust, burn time)

3. **Set Environment** (Sidebar):
   - Launch site elevation
   - Rail length
   - Launch angle (degrees from vertical)
   - Heading (degrees, 0=North)

4. **Run Simulation**:
   - Click "🚀 Run Simulation" button
   - Wait for simulation to complete
   - View results automatically

5. **View Results**:
   - Interactive charts in multiple tabs:
     - **Trajectory**: Altitude, velocity, downrange
     - **Performance**: Key metrics, Mach number
     - **Aerodynamics**: Angle of attack, dynamic pressure
     - **3D Path**: 3D trajectory visualization
   - Download CSV data

6. **Launch Elodin Editor**:
   - Click "🎮 Launch Elodin Editor" button
   - Opens Elodin with 3D visualization and telemetry

## Rocket Components

### Default Calisto Rocket
- **Mass**: 14.426 kg (dry)
- **Length**: 2.338 m
- **Diameter**: 127 mm
- **Motor**: Cesaroni M1670
- **Parachutes**: Main (10 m²) and Drogue (1 m²)

### Custom Rocket Builder
Supports:
- Nose cones (von Karman shape)
- Body tubes
- Trapezoidal fin sets
- Motor mounts
- Parachutes (multiple with different triggers)

## Motor Configuration

### Default M1670 Motor
- **Burn Time**: 3.9 s
- **Max Thrust**: 2200 N
- **Avg Thrust**: 1545 N
- **Total Impulse**: 6026 N·s

### Custom Motor Builder
- Simple thrust curve generation
- Configurable dimensions and mass
- Automatic impulse calculation

## Simulation Parameters

- **Max Time**: Maximum simulation duration (default: 200s)
- **Time Step**: Integration step size (default: 0.01s)
- **Rail Length**: Launch rail length (default: 5.2m)
- **Launch Angle**: Degrees from vertical (0 = straight up)
- **Heading**: Launch direction in degrees (0 = North)

## Visualization

### Charts
- Altitude vs Time
- Velocity vs Time
- Trajectory (Downrange vs Altitude)
- Mach Number vs Time
- Angle of Attack vs Time
- Dynamic Pressure vs Time
- 3D Flight Path

### Metrics
- Max Altitude
- Max Velocity
- Apogee Time
- Flight Time

## Elodin Integration

The Elodin editor provides:
- Real-time 3D visualization
- Interactive telemetry graphs
- Camera tracking
- Trajectory trail
- Full 6-DOF dynamics visualization

## Troubleshooting

### Elodin Not Available
- Make sure you're in the nix shell: `nix develop`
- Check that elodin is installed: `python -c "import elodin"`

### Simulation Errors
- Check rocket configuration for invalid values
- Ensure motor parameters are reasonable
- Verify parachute deployment settings

### Import Errors
- Reinstall dependencies: `./install_deps.sh`
- Make sure you're in the nix shell: `nix develop`
- Activate the virtual environment: `source venv/bin/activate`

## Examples

### Quick Start
1. Select "Calisto (Default)" rocket
2. Select "Cesaroni M1670 (Default)" motor
3. Click "🚀 Run Simulation"
4. View results and launch Elodin editor

### Custom Rocket
1. Select "Custom Rocket"
2. Configure components in sidebar
3. Set motor parameters
4. Run simulation
5. Compare with default Calisto

## Notes

- The simulation uses RK4 integration for numerical stability
- Parachute deployment supports APOGEE, ALTITUDE, and TIME triggers
- Custom motors use simplified thrust curves (bell curve approximation)
- For advanced motor curves, modify the `build_custom_motor` function

