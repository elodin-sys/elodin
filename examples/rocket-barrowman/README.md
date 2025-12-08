# Rocket Flight Simulator with AI Design

A complete rocket simulation system that lets you design rockets using natural language, run high-fidelity 6-DOF flight simulations, and visualize results in 3D. Perfect for students, engineers, and rocket enthusiasts.

## What is This?

This is a **rocket flight simulator** that includes:

1. **Web Interface (Streamlit)**: Easy-to-use web app for designing and simulating rockets
2. **AI Rocket Builder**: Describe your rocket in plain English (e.g., "I want a 50 lb rocket that goes to 10000 ft")
3. **Real Motor Database**: Access thousands of real rocket motors from ThrustCurve.org
4. **3D Visualization**: See your rocket design and flight trajectory in beautiful 3D
5. **Elodin Integration**: Advanced visualization with real-time telemetry

## Prerequisites

Before you start, you need:

1. **Linux or macOS** (Windows support via WSL)
2. **Nix package manager** installed ([install Nix](https://nixos.org/download.html))
3. **A web browser** (Chrome, Firefox, Safari, etc.)
4. **An OpenAI API key** (optional, for AI builder - get one at [platform.openai.com](https://platform.openai.com))

## Installation (Step-by-Step)

### Step 1: Clone the Repository

If you haven't already:

```bash
git clone <repository-url>
cd elodin
```

### Step 2: Enter the Nix Shell

The Nix shell provides all the required dependencies (Python, Elodin, etc.):

```bash
nix develop
```

**What to expect:**
- Your terminal prompt may change (you might see `[nix-shell:...]`)
- This is normal - you're now in the development environment
- **Keep this terminal open** - you'll need it for the next steps

**Troubleshooting:**
- If `nix develop` fails, make sure Nix is installed: `which nix`
- If you see errors about flakes, try: `nix develop --command bash`

### Step 3: Navigate to the Rocket Simulator

```bash
cd examples/rocket-barrowman
```

### Step 4: Install Python Dependencies

We'll create a virtual environment and install required packages:

```bash
./install_deps.sh
```

**What this does:**
- Creates a Python virtual environment (`venv/`)
- Installs Streamlit, Plotly, OpenAI, and other dependencies
- Takes 1-2 minutes the first time

**What to expect:**
- You'll see installation progress
- If it asks for confirmation, type `y` or press Enter
- When done, you'll see "✅ Python dependencies installed."

**Troubleshooting:**
- If `./install_deps.sh` doesn't work, make it executable: `chmod +x install_deps.sh`
- If you see "Not in nix shell", go back to Step 2

### Step 5: Run the Application

```bash
./run_streamlit.sh
```

**What to expect:**
- Streamlit will start and show a URL like `http://localhost:8501`
- Your browser should automatically open
- If not, copy the URL and paste it in your browser

**You're ready!** The web interface should now be open in your browser.

## First Time Using the App

### Option 1: Try the AI Builder (Recommended for Beginners)

1. In the sidebar, select **"AI Builder"** as the rocket type
2. In the text box, type something like:
   ```
   I want a rocket that goes to 10000 feet
   ```
3. Click **"✨ Generate Rocket Design"**
4. The AI will design a complete rocket for you!
5. Scroll down to see:
   - Rocket dimensions and mass
   - Motor selection
   - 3D visualization
   - Manufacturing specs

### Option 2: Use the Default Rocket

1. In the sidebar, select **"Calisto (Default)"** as the rocket type
2. Select **"Cesaroni M1670 (Default)"** as the motor
3. Scroll down and click **"🚀 Run Simulation"**
4. Wait a few seconds for the simulation to complete
5. View the results and charts

### Option 3: Build a Custom Rocket

1. Select **"Custom Rocket"** as the rocket type
2. Configure each component in the sidebar:
   - Nose cone (length, material)
   - Body tube (diameter, length)
   - Fins (count, size)
   - Motor mount
   - Parachutes
3. Click **"🚀 Run Simulation"** when ready

## Understanding the Interface

### Sidebar (Left Side)

- **Rocket Type**: Choose AI Builder, Custom Rocket, or Default
- **Motor Configuration**: Select or create a motor
- **Environment**: Launch site settings (elevation, angle, etc.)
- **Simulation Parameters**: Time step, max time, etc.

### Main Area (Center)

- **Rocket Design Summary**: Overview of your rocket
- **3D Visualization**: Interactive 3D view of your rocket
- **Simulation Results**: Charts and graphs after running
- **Actions**: Buttons to view results and launch Elodin

## Key Features Explained

### AI Rocket Builder

**What it does:** Converts natural language into a complete rocket design.

**Example inputs:**
- "I want a 50 lb rocket that goes to 10000 ft"
- "Build a rocket for a 6U cubesat weighing 10 lbs"
- "Rocket to 20000 ft with 4 fins and 2 kg payload"

**What you get:**
- Complete rocket design with all components
- Motor selection from real database
- Mass calculations
- Manufacturing specifications
- Bill of materials (BOM)
- Space claims for all components

### Motor Database

**What it does:** Downloads real rocket motor data from ThrustCurve.org.

**How to use:**
1. In the sidebar, find "Motor Database" section
2. Click "Download/Update Motors"
3. Wait for download (first time takes a few minutes)
4. Motors are now cached locally for fast access

**What you get:**
- Access to thousands of real motors
- Search by impulse class (A through O)
- Real thrust curves
- Motor specifications

### 3D Visualization

**What it shows:**
- Your rocket design in 3D
- All components (nose, body, fins, motor, parachutes)
- Internal components (payload bay, avionics bay)
- Packed parachutes in their bays

**How to use:**
- Rotate: Click and drag
- Zoom: Scroll wheel
- Pan: Right-click and drag

### Elodin Editor

**What it is:** Advanced 3D visualization with real-time telemetry.

**How to use:**
1. Run a simulation first
2. Click "🎮 Launch Elodin Editor"
3. A new window opens with:
   - 3D flight visualization
   - Real-time telemetry graphs
   - Camera tracking the rocket
   - Trajectory trail

**Note:** Make sure you're in the nix shell for this to work.

## Common Tasks

### Running Your First Simulation

1. Open the app (see Installation Step 5)
2. Select "Calisto (Default)" rocket
3. Select "Cesaroni M1670 (Default)" motor
4. Click "🚀 Run Simulation"
5. Wait for results
6. Explore the charts and graphs

### Designing a Custom Rocket

1. Select "Custom Rocket"
2. Configure nose cone:
   - Length: 0.3 m
   - Material: Fiberglass
3. Configure body tube:
   - Length: 1.5 m
   - Radius: 0.0635 m (2.5 inches)
4. Configure fins:
   - Count: 4
   - Root chord: 0.15 m
5. Add parachutes:
   - Main: 2.0 m diameter
   - Drogue: 0.5 m diameter
6. Run simulation

### Using the AI Builder

1. Select "AI Builder"
2. Enter your requirements:
   ```
   I want a rocket that goes to 5000 meters, 
   carries a 3U cubesat weighing 5 kg, 
   and has 4 fins
   ```
3. Click "Generate Rocket Design"
4. Review the design
5. Adjust if needed
6. Run simulation

### Downloading Motor Data

1. In sidebar, scroll to "Motor Database"
2. Click "Download/Update Motors"
3. Wait for download (shows progress)
4. Once complete, motors are available in dropdowns

## Understanding the Results

After running a simulation, you'll see:

### Charts

- **Trajectory**: Altitude vs time, velocity vs time
- **Performance**: Max altitude, max velocity, flight time
- **Aerodynamics**: Angle of attack, dynamic pressure
- **3D Path**: Flight path in 3D space

### Key Metrics

- **Max Altitude**: Highest point reached (meters/feet)
- **Max Velocity**: Fastest speed (m/s)
- **Apogee Time**: Time to reach highest point (seconds)
- **Flight Time**: Total time until landing (seconds)

### What the Numbers Mean

- **Altitude**: How high the rocket goes
- **Velocity**: How fast it's moving
- **Mach Number**: Speed relative to sound (1.0 = speed of sound)
- **Angle of Attack**: Angle between rocket and flight direction
- **Dynamic Pressure**: Air pressure on the rocket

## Troubleshooting

### "Not in nix shell" Error

**Problem:** Scripts say you're not in the nix shell.

**Solution:**
```bash
cd /home/kush-mahajan/elodin
nix develop
cd examples/rocket-barrowman
```

### Streamlit Won't Start

**Problem:** `./run_streamlit.sh` fails or browser doesn't open.

**Solutions:**
1. Make sure you're in nix shell: `echo $IN_NIX_SHELL` (should show a path)
2. Check if dependencies are installed: `ls venv/` (should exist)
3. Try running manually:
   ```bash
   source venv/bin/activate
   streamlit run app.py
   ```

### Elodin Editor Won't Launch

**Problem:** Clicking "Launch Elodin Editor" does nothing.

**Solutions:**
1. Make sure you ran a simulation first (creates data files)
2. Make sure you're in nix shell: `nix develop`
3. Try running manually:
   ```bash
   cd examples/rocket-barrowman
   elodin editor main.py
   ```

### AI Builder Not Working

**Problem:** AI builder says it needs OpenAI API key.

**Solution:**
1. Get an API key from [platform.openai.com](https://platform.openai.com)
2. In the Streamlit app, go to "OpenAI API Configuration" section
3. Paste your API key
4. Or set environment variable: `export OPENAI_API_KEY=your-key-here`

### Motors Not Loading

**Problem:** Motor database is empty or download fails.

**Solutions:**
1. Check internet connection
2. Click "Clear Cache" then "Download/Update Motors" again
3. Check ThrustCurve.org is accessible: `curl https://www.thrustcurve.org`

### Import Errors

**Problem:** Python can't find modules.

**Solution:**
```bash
# Make sure you're in nix shell
nix develop

# Reinstall dependencies
cd examples/rocket-barrowman
rm -rf venv
./install_deps.sh
```

## Project Structure

```
rocket-barrowman/
├── README.md                 # This file
├── app.py                    # Streamlit web interface
├── ai_rocket_builder.py     # AI-powered rocket design
├── motor_scraper.py          # ThrustCurve.org API client
├── rocket_visualizer.py      # 3D/2D visualization
├── flight_solver.py          # 6-DOF physics engine
├── rocket_model.py           # Rocket properties
├── motor_model.py            # Motor simulation
├── environment.py            # Atmosphere models
├── calisto_builder.py        # Default rocket definition
├── main.py                   # Elodin integration
├── requirements.txt          # Python dependencies
├── install_deps.sh           # Setup script
├── run_streamlit.sh          # Launcher script
└── STREAMLIT_README.md       # Detailed UI documentation
```

## What Each File Does

- **`app.py`**: The main web interface - this is what you interact with
- **`ai_rocket_builder.py`**: Understands natural language and designs rockets
- **`motor_scraper.py`**: Downloads motor data from ThrustCurve.org
- **`rocket_visualizer.py`**: Creates 3D and 2D visualizations
- **`flight_solver.py`**: The physics engine that simulates flight
- **`rocket_model.py`**: Calculates rocket mass, center of gravity, etc.
- **`motor_model.py`**: Simulates motor thrust and mass flow
- **`environment.py`**: Models air density, pressure, temperature
- **`main.py`**: Connects simulation to Elodin for advanced visualization

## Next Steps

1. **Try the AI Builder**: Experiment with different requirements
2. **Explore Motor Database**: Download motors and see what's available
3. **Design Custom Rockets**: Build your own from scratch
4. **Run Simulations**: See how different designs perform
5. **Visualize in Elodin**: Use the advanced 3D visualization

## Getting Help

- **Detailed UI Guide**: See `STREAMLIT_README.md`
- **AI Builder Guide**: See `AI_BUILDER_README.md`
- **API Integration**: See `API_INTEGRATION.md`
- **Issues**: Check the troubleshooting section above

## Technical Details (For Advanced Users)

### Physics

The simulator uses:
- **6-DOF Dynamics**: Full 3D motion with rotation
- **Barrowman Method**: Aerodynamic calculations
- **RK4 Integration**: Numerical solver for stability
- **ISA Atmosphere**: Standard atmospheric model

### Rocket Components

Supports:
- Nose cones (von Karman, ogive, etc.)
- Body tubes
- Fins (trapezoidal, elliptical)
- Motor mounts
- Parachutes (multiple with triggers)
- Payload bays
- Avionics bays

### Motor Data

- Source: ThrustCurve.org API
- Format: Real thrust curves from manufacturers
- Caching: Local cache for offline use
- Updates: Can refresh from API

## License

Part of the Elodin project.
