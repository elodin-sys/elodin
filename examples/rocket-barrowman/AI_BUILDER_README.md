# AI Rocket Builder & Motor Scraper

## Overview

The rocket simulator now includes two powerful new features:

1. **AI/Heuristic Rocket Builder**: Automatically designs rockets from natural language requirements
2. **Motor Scraper**: Downloads motor data from thrustcurve.org

## AI Rocket Builder

### Usage

In the Streamlit app, select "AI Builder" as the rocket type, then describe your requirements in natural language.

### Examples

```
"I want a rocket that goes to 10000 ft, carries a 6U payload that weighs 10 lbs"
```

```
"Build me a rocket for a 3U cubesat weighing 5 kg to reach 5000 meters"
```

```
"Rocket to 20000 ft with 2 kg payload, max diameter 6 inches"
```

### Supported Requirements

The AI builder can extract:

- **Target Altitude**: "goes to 10000 ft", "reach 5000 meters", "altitude of 3 km"
- **Payload Mass**: "weighs 10 lbs", "5 kg payload", "weight of 2.5 kg"
- **Payload Size**: "6U", "3U", "1U" (cubesat format)
- **Diameter Constraint**: "max diameter 6 inches", "diameter of 127mm"
- **Length Constraint**: "max length 2 meters"
- **Recovery Method**: "dual deploy", "parachute"
- **Motor Preference**: "M class motor", "L1670"

### How It Works

1. **Natural Language Parsing**: Uses regex patterns to extract requirements from text
2. **Rocket Sizing**: Calculates dimensions based on payload and altitude requirements
3. **Motor Selection**: Estimates required impulse and selects appropriate motor from database
4. **Parachute Sizing**: Calculates parachute sizes for safe recovery
5. **Component Design**: Designs nose cone, body tube, fins, and motor mount

### Design Heuristics

- **Nose Cone**: 2:1 length-to-diameter ratio (standard for von Karman)
- **Body Diameter**: Sized to payload or standard sizes (38mm, 54mm, 75mm, 98mm, 127mm)
- **Fins**: 1.5x body radius span, 4 fins standard
- **Motor Selection**: Chooses motor with impulse closest to requirement (with 20% margin)
- **Parachutes**: 
  - Drogue: Sized for 30-50 m/s descent at high altitude
  - Main: Sized for 4-6 m/s descent at low altitude

## Motor Scraper

### Features

- Scrapes motor data from thrustcurve.org
- Extracts thrust curves, specifications, and metadata
- Caches results locally for faster access
- Supports multiple motor formats (.eng, .rse, CSV)

### Usage

1. **Load Existing Database**: Click "📥 Load Motor Database" to load previously scraped motors
2. **Scrape New Motors**: Click "🌐 Scrape Motors from ThrustCurve" to download motors
3. **Automatic Integration**: Scraped motors are automatically available to the AI builder

### Motor Database Format

Motors are stored in JSON format with:
- Designation (e.g., "M1670")
- Manufacturer
- Dimensions (diameter, length)
- Mass properties (total, propellant, case)
- Thrust curve (time, thrust pairs)
- Performance metrics (impulse, burn time, max/avg thrust)

### Scraping Process

1. Searches thrustcurve.org for motors by class (M, L, K, J, etc.)
2. Scrapes individual motor pages
3. Extracts specifications from HTML tables
4. Downloads and parses thrust curve data files
5. Caches results in `motor_cache/` directory

### Rate Limiting

The scraper includes 1-second delays between requests to be respectful to the server.

## Integration

### AI Builder + Motor Database

When motors are loaded, the AI builder will:
1. Select the best motor from the database based on requirements
2. Use actual thrust curves instead of approximations
3. Provide more accurate performance estimates

### Workflow

1. **First Time**: Scrape motors to build database
2. **Design Rocket**: Use AI builder with natural language
3. **Review Design**: Check generated configuration
4. **Run Simulation**: Test the design
5. **Iterate**: Adjust requirements and regenerate

## Technical Details

### AI Builder Components

- `ai_rocket_builder.py`: Main builder logic
  - `RocketRequirements`: Parsed requirements dataclass
  - `RocketDesigner`: Design algorithm
  - `parse_requirements()`: NLP parsing
  - `design_rocket()`: Rocket sizing
  - `build_rocket_config()`: Configuration generation

### Motor Scraper Components

- `motor_scraper.py`: Scraper implementation
  - `ThrustCurveScraper`: Main scraper class
  - `MotorData`: Motor data structure
  - `scrape_motor_page()`: Individual motor scraping
  - `scrape_motor_list()`: Batch scraping
  - `save_motor_database()`: Database persistence

### File Structure

```
rocket-barrowman/
├── app.py                    # Streamlit UI with AI builder integration
├── ai_rocket_builder.py      # AI/heuristic rocket designer
├── motor_scraper.py          # ThrustCurve.org scraper
├── motor_cache/              # Cached motor data
│   └── motor_database.json   # Motor database
└── requirements.txt          # Dependencies (includes beautifulsoup4, requests)
```

## Limitations

### AI Builder

- Uses simplified heuristics (not full physics simulation)
- Motor selection is approximate (actual performance may vary)
- Parachute sizing is conservative (may be oversized)
- Limited to standard rocket configurations

### Motor Scraper

- Depends on thrustcurve.org website structure (may break if site changes)
- Some motors may not have complete data
- Thrust curve parsing may need adjustment for different formats
- Rate limiting may make large scrapes slow

## Future Improvements

1. **Better NLP**: Use actual NLP models (spaCy, transformers) for better parsing
2. **Physics-Based Sizing**: Use iterative simulation to optimize design
3. **Multi-Objective Optimization**: Optimize for cost, performance, reliability
4. **Motor Database API**: Direct API access instead of scraping
5. **Design Validation**: Check stability, structural integrity automatically
6. **3D Preview**: Show generated rocket design before simulation

## Examples

### Example 1: High-Altitude Payload Rocket

**Input**: "I need a rocket to carry a 6U cubesat weighing 8 kg to 15000 feet"

**Output**:
- Rocket length: ~2.5 m
- Diameter: 127 mm (to fit 6U payload)
- Motor: M-class or larger
- Dual-deploy recovery
- Estimated apogee: ~4500 m

### Example 2: Small Educational Rocket

**Input**: "Build a simple rocket for a 1U payload to reach 3000 meters"

**Output**:
- Rocket length: ~1.5 m
- Diameter: 75 mm
- Motor: J or K class
- Single parachute recovery
- Estimated apogee: ~3000 m

## Troubleshooting

### AI Builder Not Working

- Check that requirements are clear and include key parameters
- Try rephrasing with explicit units (ft, m, kg, lbs)
- Ensure payload size is specified if using cubesat format

### Motor Scraper Errors

- Check internet connection
- Verify thrustcurve.org is accessible
- Some motors may have incomplete data (this is normal)
- Try scraping smaller batches if timeouts occur

### Motor Database Empty

- Run scraper first to populate database
- Check `motor_cache/` directory exists
- Verify JSON file is valid if manually editing

