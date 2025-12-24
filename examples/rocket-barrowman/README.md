# Rocket Flight Simulator with AI Design

A complete rocket simulation system that lets you design rockets using natural language, run high-fidelity 6-DOF flight simulations, and visualize results in 3D.

## Features

- **Web Interface (Streamlit)**: Easy-to-use web app for designing and simulating rockets
- **AI Rocket Builder**: Describe your rocket in plain English (e.g., "I want a 50 lb rocket that goes to 10000 ft")
- **Real Motor Database**: Access thousands of real rocket motors from ThrustCurve.org
- **3D Visualization**: See your rocket design and flight trajectory in beautiful 3D
- **Elodin Integration**: Advanced visualization with real-time telemetry

## Quickstart

```bash
# Clone and enter repository
git clone git@github.com:elodin-sys/elodin.git && cd elodin

# Enter nix shell and install elodin
nix develop
just install

# Set up Python environment
uv venv --python 3.12 && source .venv/bin/activate
uvx maturin develop --uv --manifest-path=libs/nox-py/Cargo.toml
uv pip install -r examples/rocket-barrowman/requirements.txt

# Run the Streamlit web interface
examples/rocket-barrowman/run_streamlit.sh

# Or run the Elodin editor directly
elodin editor examples/rocket-barrowman/main.py
```

## Using the App

There are three ways to create and simulate rockets:

### Option 1: AI Builder (Recommended for Beginners)

The AI Builder converts natural language into complete rocket designs.

1. In the sidebar, select **"AI Builder"** as the rocket type
2. In the text box, describe your rocket:
   ```
   I want a rocket that goes to 10000 feet
   ```
   Or be more specific:
   ```
   Build a 50 lb rocket with 4 fins that carries a 2 kg payload to 5000 meters
   ```
3. Click **"✨ Generate Rocket Design"**
4. The AI will design a complete rocket including:
   - Rocket dimensions and mass
   - Motor selection from the real motor database
   - 3D visualization
   - Manufacturing specifications
   - Bill of materials

**Note:** Requires an OpenAI API key. Get one at [platform.openai.com](https://platform.openai.com), then enter it in the app's sidebar or set `OPENAI_API_KEY` environment variable.

### Option 2: Default Rocket (Quick Demo)

Use the pre-configured Calisto rocket for a quick simulation:

1. In the sidebar, select **"Calisto (Default)"** as the rocket type
2. Select **"Cesaroni M1670 (Default)"** as the motor
3. Optionally adjust environment settings (elevation, launch angle, heading)
4. Click **"🚀 Run Simulation"**
5. View results: altitude, velocity, trajectory charts, and 3D flight path

### Option 3: Custom Rocket (Full Control)

Build your own rocket from scratch:

1. Select **"Custom Rocket"** as the rocket type
2. Configure each component in the sidebar:
   - **Nose cone**: Length, thickness, material (fiberglass, carbon fiber, etc.)
   - **Body tube**: Length, radius, wall thickness, material
   - **Fins**: Count, root chord, tip chord, span, sweep, material
   - **Motor mount**: Dimensions and position
   - **Parachutes**: Main and drogue with deployment triggers (apogee, altitude, time)
3. Select or configure a motor
4. Click **"🚀 Run Simulation"**

## Viewing Results

After running a simulation, you'll see:

- **Trajectory charts**: Altitude vs time, velocity vs time, downrange distance
- **Performance metrics**: Max altitude, max velocity, apogee time, flight duration
- **Aerodynamics**: Angle of attack, dynamic pressure, Mach number
- **3D flight path**: Interactive 3D visualization of the trajectory

Click **"🎮 Launch Elodin Editor"** for advanced 3D visualization with real-time telemetry graphs.

## Default Configuration

**Calisto Rocket:** 14.426 kg dry mass, 2.338 m length, 127 mm diameter, main + drogue parachutes

**M1670 Motor:** 3.9s burn time, 2200 N max thrust, 1545 N avg thrust, 6026 N·s total impulse

**Simulation:** 200s max time, 0.01s time step, 5.2m rail length

## Troubleshooting

### Streamlit Won't Start

```bash
# Make sure you're in the activated venv
source .venv/bin/activate

# Try running manually
cd examples/rocket-barrowman
streamlit run app.py
```

### Elodin Editor Won't Launch

```bash
# Run from repo root with venv activated
elodin editor examples/rocket-barrowman/main.py
```

### Import Errors

```bash
# Reinstall dependencies
uv pip install -r examples/rocket-barrowman/requirements.txt
```

## Project Structure

```
rocket-barrowman/
├── README.md                 # This file
├── main.py                   # Elodin integration entry point
├── calisto_builder.py        # Default rocket definition
├── requirements.txt          # Python dependencies
├── run_streamlit.sh          # Launcher script
│
├── core/                     # Core physics and modeling
│   ├── __init__.py
│   ├── flight_solver.py      # 6-DOF physics engine
│   ├── rocket_model.py       # Rocket properties
│   ├── motor_model.py        # Motor simulation
│   ├── environment.py        # Atmosphere models
│   ├── math_utils.py         # Math utilities
│   ├── motor_scraper.py      # ThrustCurve.org API client
│   └── openrocket_*.py       # OpenRocket compatibility layer
│
├── ui/                       # User interface
│   ├── __init__.py
│   ├── app.py                # Streamlit web interface
│   ├── rocket_visualizer.py  # 3D/2D visualization
│   ├── rocket_renderer.py    # Plotly-based renderer
│   └── mesh_renderer.py     # Trimesh-based 3D mesh generation
│
├── optimization/             # AI-powered design optimization
│   ├── __init__.py
│   ├── smart_optimizer.py    # Advanced iterative optimizer
│   ├── ai_rocket_builder.py  # NLP-based rocket designer
│   └── ai_rocket_optimizer.py # Legacy optimizer
│
├── analysis/                 # Flight analysis and metrics
│   ├── __init__.py
│   └── flight_analysis.py   # Aerospace-grade analysis suite
│
└── docs/                     # Documentation
    ├── WHITEPAPER.md         # Technical whitepaper
    ├── AI_BUILDER_README.md   # AI Builder guide
    ├── API_INTEGRATION.md    # API integration guide
    └── sources/              # Source materials
        └── barrowman_equation.tex
```

## Technical Details

### Physics

- **6-DOF Dynamics**: Full 3D motion with rotation
- **Barrowman Method**: Aerodynamic calculations for stability and drag
- **RK4 Integration**: Numerical solver for stability
- **ISA Atmosphere**: Standard atmospheric model

### Supported Components

- Nose cones (von Karman, ogive, conical, etc.)
- Body tubes
- Fins (trapezoidal, elliptical)
- Motor mounts
- Parachutes (APOGEE, ALTITUDE, or TIME triggers)
- Payload and avionics bays

### Motor Data

- Source: ThrustCurve.org API
- Caching: Local cache for offline use

## Additional Documentation

- **Technical Whitepaper**: See `docs/WHITEPAPER.md`
- **AI Builder Guide**: See `docs/AI_BUILDER_README.md`
- **API Integration**: See `docs/API_INTEGRATION.md`
- **Barrowman Equations**: See `docs/sources/barrowman_equation.tex`

## License

Part of the Elodin project.
