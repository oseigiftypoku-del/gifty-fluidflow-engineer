"""
Gifty Fluidflow Engineer - Calculation Engine
Contains core fluid mechanics, friction factor, pressure drop, head loss,
and Bernoulli/energy analysis functions.
"""

import math
from typing import Dict, Any, Tuple, Optional


def calculate_cross_sectional_area(diameter_m: float) -> float:
    """Calculate circular pipe cross-sectional area A = pi * D^2 / 4 [m^2]."""
    if diameter_m <= 0:
        raise ValueError("Pipe diameter must be greater than zero.")
    return (math.pi * (diameter_m ** 2)) / 4.0


def calculate_velocity_from_flow(flow_rate_m3_s: float, diameter_m: float) -> float:
    """Calculate mean velocity v = Q / A [m/s]."""
    area = calculate_cross_sectional_area(diameter_m)
    return flow_rate_m3_s / area


def calculate_flow_from_velocity(velocity_m_s: float, diameter_m: float) -> float:
    """Calculate volumetric flow rate Q = v * A [m^3/s]."""
    area = calculate_cross_sectional_area(diameter_m)
    return velocity_m_s * area


def calculate_mass_flow_rate(density_kg_m3: float, flow_rate_m3_s: float) -> float:
    """Calculate mass flow rate m_dot = rho * Q [kg/s]."""
    return density_kg_m3 * flow_rate_m3_s


def calculate_reynolds_number(
    density_kg_m3: float,
    velocity_m_s: float,
    diameter_m: float,
    dynamic_viscosity_pa_s: float,
) -> float:
    """
    Calculate Reynolds number Re = (rho * v * D) / mu [-].
    Equivalently Re = v * D / nu.
    """
    if dynamic_viscosity_pa_s <= 0:
        raise ValueError("Dynamic viscosity must be positive.")
    if density_kg_m3 <= 0:
        raise ValueError("Density must be positive.")
    if diameter_m <= 0:
        raise ValueError("Diameter must be positive.")
    
    return (density_kg_m3 * abs(velocity_m_s) * diameter_m) / dynamic_viscosity_pa_s


def classify_flow_regime(reynolds: float) -> Tuple[str, str]:
    """
    Classify internal pipe flow regime based on standard criteria:
    - Re < 2300: Laminar
    - 2300 <= Re < 4000: Transitional
    - Re >= 4000: Turbulent
    """
    if reynolds < 2300:
        return (
            "Laminar",
            "Viscous forces dominate inertial forces. Streamlines remain orderly with a parabolic velocity profile."
        )
    elif reynolds < 4000:
        return (
            "Transitional",
            "Flow alternates unpredictably between laminar and turbulent characteristics; boundary layer experiences intermittent burst instability."
        )
    else:
        return (
            "Turbulent",
            "Inertial forces dominate viscous forces. Chaotic eddy mixing, flattened velocity profile, and increased shear stress occur."
        )


def calculate_swamee_jain_friction_factor(reynolds: float, relative_roughness: float) -> float:
    """
    Calculate explicit Swamee-Jain approximation for Darcy friction factor:
    f = 0.25 / [log10( (epsilon / (3.7 * D)) + (5.74 / Re^0.9) )]^2
    Valid for 5000 <= Re <= 10^8 and 1e-6 <= epsilon/D <= 1e-2.
    """
    if reynolds <= 0:
        raise ValueError("Reynolds number must be positive for friction factor calculation.")
    
    term = (relative_roughness / 3.7) + (5.74 / (reynolds ** 0.9))
    if term <= 0:
        term = 1e-8
    
    return 0.25 / ((math.log10(term)) ** 2)


def solve_colebrook_white_friction_factor(
    reynolds: float,
    relative_roughness: float,
    tol: float = 1e-7,
    max_iter: int = 50,
) -> Tuple[float, int, bool]:
    """
    Solve implicit Colebrook-White equation for turbulent Darcy friction factor:
    1 / sqrt(f) = -2 * log10( (epsilon / (3.7 * D)) + (2.51 / (Re * sqrt(f))) )

    Uses Newton-Raphson method with Swamee-Jain initial estimate.
    Returns (friction_factor, iterations_used, converged_bool).
    """
    if reynolds < 2300:
        # Laminar flow exact formula
        return (64.0 / max(reynolds, 1e-5), 1, True)

    # Initial guess from Swamee-Jain
    try:
        f_guess = calculate_swamee_jain_friction_factor(reynolds, relative_roughness)
    except Exception:
        f_guess = 0.02

    f = max(0.005, min(0.15, f_guess))
    rr_term = relative_roughness / 3.7

    for iteration in range(1, max_iter + 1):
        sqrt_f = math.sqrt(f)
        arg = rr_term + (2.51 / (reynolds * sqrt_f))
        if arg <= 0:
            arg = 1e-10

        # Residual: F(f) = 1/sqrt(f) + 2*log10(arg) = 0
        F = (1.0 / sqrt_f) + 2.0 * math.log10(arg)

        # Derivative: dF/df = -0.5 * f^(-1.5) - (2 / (ln(10) * arg)) * (2.51 / Re) * (-0.5 * f^(-1.5))
        ln10 = math.log(10.0)
        dF_df = -0.5 * (f ** -1.5) * (1.0 - (2.0 / ln10) * (2.51 / (reynolds * arg)))

        if abs(dF_df) < 1e-12:
            break

        f_next = f - (F / dF_df)

        # Dampen oscillations and maintain physical bounds
        if f_next <= 0.002 or f_next > 0.2:
            f_next = 0.5 * (f + max(0.002, min(0.2, f_next)))

        if abs(f_next - f) < tol:
            return (f_next, iteration, True)

        f = f_next

    # Fallback to Swamee-Jain if Newton-Raphson fails to reach tolerance
    return (f, max_iter, False)


def calculate_darcy_friction_factor(
    reynolds: float,
    relative_roughness: float,
) -> Dict[str, Any]:
    """
    Calculate comprehensive friction factors across all regimes:
    - Laminar: f = 64/Re
    - Transitional: Weighted cubic interpolation
    - Turbulent: Colebrook-White and Swamee-Jain comparison
    """
    if reynolds < 2300:
        f_lam = 64.0 / max(reynolds, 1e-5)
        return {
            "friction_factor": f_lam,
            "colebrook_f": f_lam,
            "swamee_jain_f": f_lam,
            "method": "Laminar Exact (f = 64/Re)",
            "iterations": 1,
            "converged": True,
            "abs_difference": 0.0,
            "pct_difference": 0.0,
        }
    elif reynolds < 4000:
        # Transitional regime interpolation
        f_2300 = 64.0 / 2300.0
        f_turb_4000, iters, conv = solve_colebrook_white_friction_factor(4000.0, relative_roughness)
        # Linear interpolation parameter t in [0, 1]
        t = (reynolds - 2300.0) / (4000.0 - 2300.0)
        f_trans = (1.0 - t) * f_2300 + t * f_turb_4000
        f_sj = calculate_swamee_jain_friction_factor(reynolds, relative_roughness)

        return {
            "friction_factor": f_trans,
            "colebrook_f": f_trans,
            "swamee_jain_f": f_sj,
            "method": "Transitional Interpolation (Laminar 2300 to Turbulent 4000)",
            "iterations": iters,
            "converged": conv,
            "abs_difference": abs(f_trans - f_sj),
            "pct_difference": (abs(f_trans - f_sj) / f_trans) * 100.0 if f_trans > 0 else 0.0,
        }
    else:
        f_cw, iters, conv = solve_colebrook_white_friction_factor(reynolds, relative_roughness)
        f_sj = calculate_swamee_jain_friction_factor(reynolds, relative_roughness)
        abs_diff = abs(f_cw - f_sj)
        pct_diff = (abs_diff / f_cw) * 100.0 if f_cw > 0 else 0.0

        return {
            "friction_factor": f_cw,
            "colebrook_f": f_cw,
            "swamee_jain_f": f_sj,
            "method": "Colebrook-White (Iterative Solution)",
            "iterations": iters,
            "converged": conv,
            "abs_difference": abs_diff,
            "pct_difference": pct_diff,
        }


def calculate_pressure_drop(
    friction_factor: float,
    length_m: float,
    diameter_m: float,
    density_kg_m3: float,
    velocity_m_s: float,
) -> float:
    """
    Calculate Darcy-Weisbach frictional pressure drop:
    Delta P = f * (L / D) * (rho * v^2 / 2) [Pa]
    """
    if diameter_m <= 0:
        raise ValueError("Pipe diameter must be positive.")
    dynamic_pressure = 0.5 * density_kg_m3 * (velocity_m_s ** 2)
    return friction_factor * (length_m / diameter_m) * dynamic_pressure


def calculate_head_loss(
    friction_factor: float,
    length_m: float,
    diameter_m: float,
    velocity_m_s: float,
    gravity_m_s2: float = 9.80665,
) -> float:
    """
    Calculate Darcy-Weisbach head loss:
    h_f = f * (L / D) * (v^2 / (2 * g)) [m of fluid]
    """
    if diameter_m <= 0 or gravity_m_s2 <= 0:
        raise ValueError("Diameter and gravity must be positive.")
    return friction_factor * (length_m / diameter_m) * ((velocity_m_s ** 2) / (2.0 * gravity_m_s2))


def calculate_hydrostatic_pressure(
    density_kg_m3: float,
    elevation_diff_m: float,
    gravity_m_s2: float = 9.80665,
    atmospheric_pressure_pa: float = 101325.0,
) -> Dict[str, float]:
    """
    Calculate hydrostatic pressure:
    P_gauge = rho * g * delta_z [Pa]
    P_absolute = P_gauge + P_atm [Pa]
    """
    p_gauge = density_kg_m3 * gravity_m_s2 * elevation_diff_m
    p_abs = p_gauge + atmospheric_pressure_pa
    return {
        "gauge_pressure_pa": p_gauge,
        "absolute_pressure_pa": p_abs,
        "atmospheric_pressure_pa": atmospheric_pressure_pa,
    }


def calculate_bernoulli_energy_balance(
    p1_pa: float,
    v1_m_s: float,
    z1_m: float,
    v2_m_s: float,
    z2_m: float,
    head_loss_m: float,
    density_kg_m3: float,
    gravity_m_s2: float = 9.80665,
) -> Dict[str, float]:
    """
    Solve steady extended Bernoulli/energy equation for outlet pressure P2:
    P1/(rho*g) + v1^2/(2g) + z1 = P2/(rho*g) + v2^2/(2g) + z2 + h_L

    Rearranged:
    P2 = P1 + 0.5*rho*(v1^2 - v2^2) + rho*g*(z1 - z2) - rho*g*h_L
    """
    gamma = density_kg_m3 * gravity_m_s2
    h1_total = (p1_pa / gamma) + ((v1_m_s ** 2) / (2.0 * gravity_m_s2)) + z1_m
    p2_pa = p1_pa + (0.5 * density_kg_m3 * ((v1_m_s ** 2) - (v2_m_s ** 2))) + (gamma * (z1_m - z2_m)) - (gamma * head_loss_m)
    h2_total = (p2_pa / gamma) + ((v2_m_s ** 2) / (2.0 * gravity_m_s2)) + z2_m
    
    return {
        "inlet_pressure_pa": p1_pa,
        "outlet_pressure_pa": p2_pa,
        "total_head_inlet_m": h1_total,
        "total_head_outlet_m": h2_total,
        "elevation_head_diff_m": z1_m - z2_m,
        "velocity_head_diff_m": ((v1_m_s ** 2) - (v2_m_s ** 2)) / (2.0 * gravity_m_s2),
        "friction_head_loss_m": head_loss_m,
    }
