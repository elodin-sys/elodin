# API Integration Guide

## Overview

The rocket simulator now uses official APIs instead of web scraping:

1. **ThrustCurve.org API**: Official REST API for motor data
2. **OpenAI API**: For natural language understanding in AI builder

## ThrustCurve.org API

### Documentation
- API Documentation: https://www.thrustcurve.org/info/api.html
- Swagger Spec: https://www.thrustcurve.org/api/v1/swagger.json

### Features
- **Search API**: Find motors by various criteria
- **Download API**: Get motor data files and parsed samples
- **Metadata API**: Get available search criteria
- **Motor Guide API**: Find motors for specific rockets

### Implementation

The `ThrustCurveScraper` class now uses the API:

```python
scraper = ThrustCurveScraper()

# Search for motors
motors = scraper.search_motors(
    query="M1670",
    max_results=10,
    impulseClass="M",
    availability="available"
)

# Get detailed motor data
motor_data = scraper.get_motor_data(motor_id)
```

### API Endpoints Used

- `/api/v1/search.json` - Search for motors
- `/api/v1/download.json` - Download motor data
- `/api/v1/metadata.json` - Get search criteria

### Benefits Over Scraping

- ✅ More reliable (official API)
- ✅ Faster (direct data access)
- ✅ Structured data (JSON format)
- ✅ No HTML parsing needed
- ✅ Rate limiting handled by API
- ✅ Future-proof (API versioning)

## OpenAI API

### Setup

1. Get API key from https://platform.openai.com/api-keys
2. Set environment variable: `export OPENAI_API_KEY=sk-...`
3. Or enter in Streamlit UI under "OpenAI API Configuration"

### Usage

The AI builder automatically uses OpenAI if available:

```python
designer = RocketDesigner(motor_database, openai_api_key="sk-...")
req = designer.parse_requirements("I want a rocket to 10000 ft with 6U payload")
```

### Fallback

If OpenAI is not available, falls back to regex-based parsing.

### Benefits

- ✅ Better natural language understanding
- ✅ Handles complex requirements
- ✅ Unit conversion
- ✅ Context understanding
- ✅ More robust parsing

## Configuration

### Environment Variables

```bash
export OPENAI_API_KEY=sk-your-key-here
```

### Streamlit UI

1. Go to "AI Builder" section
2. Expand "OpenAI API Configuration"
3. Enter API key (stored in session, not permanently)

## Example Usage

### Using ThrustCurve API

```python
from motor_scraper import ThrustCurveScraper

scraper = ThrustCurveScraper()

# Search for M-class motors
results = scraper.search_motors(impulseClass="M", availability="available")

# Get full data for first result
motor = scraper.get_motor_data(results[0]['id'])

# Save to database
scraper.save_motor_database([motor])
```

### Using OpenAI API

```python
from ai_rocket_builder import RocketDesigner
import os

# Initialize with OpenAI
designer = RocketDesigner(
    motor_database=motors,
    openai_api_key=os.getenv('OPENAI_API_KEY')
)

# Parse requirements
req = designer.parse_requirements(
    "Build a rocket for a 6U cubesat weighing 10 lbs to reach 10000 feet"
)

# Generate design
config, motor = designer.build_rocket_config(req)
```

## Error Handling

### ThrustCurve API Errors

- Network errors: Retry with exponential backoff
- Rate limiting: API handles this automatically
- Invalid motor ID: Returns None
- Missing data: Falls back to defaults

### OpenAI API Errors

- API key missing: Falls back to regex parsing
- Rate limiting: OpenAI handles this
- Invalid response: Falls back to regex parsing
- Network errors: Falls back to regex parsing

## Rate Limits

### ThrustCurve.org

- No official rate limit documented
- Be respectful: 0.5s delay between requests
- Caching reduces API calls

### OpenAI

- Depends on your plan
- Free tier: Limited requests
- Paid tier: Higher limits
- See https://platform.openai.com/docs/guides/rate-limits

## Caching

Both APIs use local caching:

- Motor data cached in `motor_cache/` directory
- Cache files named by motor ID
- Reduces API calls on subsequent runs

## Migration from Scraping

The old scraping code has been removed. If you have cached data:

1. Old cache files may still work
2. New API data will overwrite old cache
3. Motor IDs changed from URLs to hex IDs

## Troubleshooting

### ThrustCurve API Not Working

- Check internet connection
- Verify API endpoint is accessible
- Check API response format hasn't changed
- Review error messages in console

### OpenAI API Not Working

- Verify API key is correct
- Check account has credits
- Review rate limits
- Check OpenAI service status

### Fallback to Regex

If OpenAI fails, the system automatically falls back to regex parsing. This is less accurate but still functional.

