"""
Calisto drag curves extracted from RocketPy plot.
These are the actual CD values used by RocketPy for validation.
"""

import numpy as np

# Extracted from the drag curve plot
# Mach, CD_power_on, CD_power_off
CALISTO_DRAG_DATA = np.array([
    [0.00, 0.340, 0.340],
    [0.05, 0.410, 0.410],
    [0.10, 0.405, 0.405],
    [0.20, 0.395, 0.395],
    [0.30, 0.390, 0.390],
    [0.40, 0.385, 0.385],
    [0.50, 0.380, 0.380],
    [0.60, 0.378, 0.378],
    [0.70, 0.405, 0.405],
    [0.80, 0.450, 0.450],
    [0.90, 0.550, 0.550],
    [1.00, 0.650, 0.650],
    [1.10, 0.750, 0.750],  # Peak transonic drag
    [1.20, 0.720, 0.720],
    [1.30, 0.680, 0.680],
    [1.40, 0.650, 0.650],
    [1.50, 0.620, 0.620],
    [1.60, 0.595, 0.595],
    [1.70, 0.570, 0.570],
    [1.80, 0.550, 0.550],
    [1.90, 0.535, 0.535],
    [2.00, 0.520, 0.520],
])


def get_calisto_cd(mach: float, power_on: bool = True) -> float:
    """
    Get Calisto's drag coefficient at given Mach number.
    
    Args:
        mach: Mach number
        power_on: If True, use power-on curve; else power-off
        
    Returns:
        Drag coefficient (dimensionless)
    """
    mach = abs(mach)
    
    # For Calisto, power on/off are the same
    cd_values = CALISTO_DRAG_DATA[:, 1]  # Both columns are identical
    mach_values = CALISTO_DRAG_DATA[:, 0]
    
    # Linear interpolation
    cd = np.interp(mach, mach_values, cd_values)
    return float(cd)


if __name__ == "__main__":
    print("Calisto Drag Curve Data:")
    print("\nMach   CD")
    print("-" * 20)
    for row in CALISTO_DRAG_DATA:
        print(f"{row[0]:5.2f}  {row[1]:6.3f}")
    
    print("\nTest interpolation:")
    test_machs = [0.0, 0.3, 0.85, 1.1, 2.0]
    for m in test_machs:
        cd = get_calisto_cd(m)
        print(f"Mach {m:4.2f}: CD = {cd:.3f}")

