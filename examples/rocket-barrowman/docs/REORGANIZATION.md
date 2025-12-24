# Project Reorganization

This document describes the reorganization of the rocket-barrowman project into a more structured directory layout.

## Changes

### Directory Structure

The project has been reorganized from a flat structure with many Python files at the root level into a modular structure:

- **`core/`** - Core physics and rocket modeling components
  - `flight_solver.py` - 6-DOF physics engine
  - `rocket_model.py` - Rocket properties
  - `motor_model.py` - Motor simulation
  - `environment.py` - Atmosphere models
  - `math_utils.py` - Math utilities
  - `motor_scraper.py` - ThrustCurve.org API client
  - `openrocket_*.py` - OpenRocket compatibility layer

- **`ui/`** - User interface components
  - `app.py` - Streamlit web interface
  - `rocket_visualizer.py` - 3D/2D visualization
  - `rocket_renderer.py` - Plotly-based renderer
  - `mesh_renderer.py` - Trimesh-based 3D mesh generation

- **`optimization/`** - AI-powered design optimization
  - `smart_optimizer.py` - Advanced iterative optimizer
  - `ai_rocket_builder.py` - NLP-based rocket designer
  - `ai_rocket_optimizer.py` - Legacy optimizer

- **`analysis/`** - Flight analysis and metrics
  - `flight_analysis.py` - Aerospace-grade analysis suite

- **`docs/`** - Documentation
  - `WHITEPAPER.md` - Technical whitepaper
  - `AI_BUILDER_README.md` - AI Builder guide
  - `API_INTEGRATION.md` - API integration guide
  - `sources/` - Source materials

### Import Updates

All imports have been updated to use the new package structure:
- Core components: `from core import ...`
- UI components: `from ui import ...`
- Optimization: `from optimization import ...`
- Analysis: `from analysis import ...`

### Benefits

1. **Better Organization** - Clear separation of concerns
2. **Easier Navigation** - Related files are grouped together
3. **Scalability** - Easier to add new features without cluttering the root
4. **Maintainability** - Clearer structure for future developers

## Migration Notes

- All existing functionality remains unchanged
- Import paths have been updated throughout
- `__init__.py` files added for proper package structure
- Entry points (`main.py`, `app.py`) updated to use new imports

## Future Enhancements

As mentioned in PR feedback:
- Consider creating an Elodin JAX simulation solver variant for Monte Carlo and parameter sweep analysis
- This would leverage Elodin's JAX integration for GPU-accelerated, differentiable physics simulations
- Would enable more advanced optimization and sensitivity analysis

