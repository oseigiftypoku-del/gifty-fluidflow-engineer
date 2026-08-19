"""
Gifty Fluidflow Engineer - Validation & Physical Feasibility Layer
Ensures all engineering inputs are non-zero, within physical limits,
and clearly advises the user when correction is required.
"""

from typing import List, Tuple, Dict, Any


def validate_fluid_flow_inputs(
    diameter_m: float,
    length_m: float,
    roughness_m: float,
    density_kg_m3: float,
    viscosity_pa_s: float,
    flow_value: float,
    flow_mode: str, # "flow_rate" or "velocity"
) -> Tuple[bool, List[str]]:
    """
    Validate fluid flow parameters in SI units.
    Returns (is_valid, list_of_error_messages).
    """
    errors: List[str] = []

    # Diameter checks
    if diameter_m <= 0:
        errors.append("Pipe inner diameter must be strictly greater than 0 (entered ≤ 0).")
    elif diameter_m < 0.0005:
        errors.append(f"Pipe diameter ({diameter_m*1000:.3f} mm) is in the microfluidic regime (< 0.5 mm); macro-scale Darcy-Weisbach assumptions may break down.")
    elif diameter_m > 10.0:
        errors.append(f"Pipe diameter ({diameter_m:.2f} m) is unusually large (> 10 m). Please verify geometry.")

    # Length checks
    if length_m <= 0:
        errors.append("Pipe length must be strictly greater than 0.")
    elif length_m > 1000000.0:
        errors.append("Pipe length exceeds 1,000 km. Please check input units.")

    # Roughness checks
    if roughness_m < 0:
        errors.append("Absolute pipe roughness (epsilon) cannot be negative.")
    elif diameter_m > 0 and roughness_m >= diameter_m:
        errors.append(f"Pipe roughness ({roughness_m*1000:.3f} mm) cannot be greater than or equal to pipe diameter ({diameter_m*1000:.3f} mm).")
    elif diameter_m > 0 and (roughness_m / diameter_m) > 0.05:
        errors.append(f"Relative roughness epsilon/D = {roughness_m/diameter_m:.4f} is exceptionally high (> 0.05), which exceeds standard Moody chart correlation bounds.")

    # Density checks
    if density_kg_m3 <= 0:
        errors.append("Fluid density (rho) must be strictly greater than 0.")
    elif density_kg_m3 > 25000.0:
        errors.append(f"Fluid density ({density_kg_m3:.1f} kg/m³) exceeds liquid mercury (~13,550 kg/m³). Please check input.")

    # Viscosity checks
    if viscosity_pa_s <= 0:
        errors.append("Dynamic viscosity (mu) must be strictly greater than 0.")
    elif viscosity_pa_s > 10000.0:
        errors.append(f"Dynamic viscosity ({viscosity_pa_s:.1f} Pa·s) is unrealistically high for fluid flow calculations.")

    # Flow condition checks
    if flow_value <= 0:
        errors.append(f"Flow {'rate' if flow_mode == 'flow_rate' else 'velocity'} must be strictly greater than 0.")
    
    if flow_mode == "velocity" and flow_value > 200.0:
        errors.append(f"Flow velocity ({flow_value:.1f} m/s) is extremely high; compressible flow / Mach number effects will dominate and Darcy-Weisbach incompressible assumption is invalid.")

    is_valid = len(errors) == 0
    return is_valid, errors
