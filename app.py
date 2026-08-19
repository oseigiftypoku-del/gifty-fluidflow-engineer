# =============================================================================
# AI-ASSISTED DEVELOPMENT DOCUMENTATION
# =============================================================================
# AI tools used:
# - ChatGPT
# - Google Gemini
#
# Key prompts used:
# 1. "Build a general-purpose Streamlit fluid flow engineering calculator
#    with Reynolds number, friction factor, pressure drop and head loss."
# 2. "Add interactive sidebar controls, Plotly visualizations and a Pandas
#    results table to the Streamlit engineering application."
# 3. "Add engineering validation, error handling, unit conversion and
#    engineering interpretation to the fluid flow calculator."
#
# Most important manual fix/verification:
# - I manually verified the engineering calculations, input validation,
#   unit conversions and deployment errors. During deployment, I identified
#   and corrected a function-name mismatch between app.py and
#   visualizations.py involving plot_moody_diagram.
# ============================================================================="""
Gifty Fluidflow Engineer
General Fluid Mechanics & Fluid Flow Engineering Calculator
Main Python Streamlit Application Entry Point

To launch:
    streamlit run app.py
"""

import math
import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Local modules
from calculations import (
    calculate_cross_sectional_area,
    calculate_velocity_from_flow,
    calculate_flow_from_velocity,
    calculate_mass_flow_rate,
    calculate_reynolds_number,
    classify_flow_regime,
    calculate_swamee_jain_friction_factor,
    solve_colebrook_white_friction_factor,
    calculate_darcy_friction_factor,
    calculate_pressure_drop,
    calculate_head_loss,
)
from fluid_properties import PREDEFINED_FLUIDS, PIPE_ROUGHNESS_PRESETS
from unit_conversion import UnitConverter
from validation import validate_fluid_flow_inputs
from visualizations import (
    plot_pressure_drop_vs_flow_rate,
    plot_moody_friction_factor_chart,
    plot_head_loss_vs_flow_rate,
)
from ai_assistant import (
    generate_engineering_explanation,
    get_gemini_api_key,
)

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Gifty Fluidflow Engineer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .main-title {
        font-family: 'Georgia', serif;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.0rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #c2a371;
        margin-top: 0px;
        margin-bottom: 1.0rem;
        font-weight: 600;
    }
    </style>
    """,unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. MAIN HEADER & GUIDE
# -----------------------------------------------------------------------------
st.markdown("<h1 class='main-title'>Gifty Fluidflow Engineer</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='sub-title'>General Fluid Mechanics & Fluid Flow Engineering Calculator</p>",
    unsafe_allow_html=True,
)

with st.expander("ℹ️ How to Use This Engineering Calculator", expanded=False):
    st.markdown(
        """
        **Welcome to Gifty Fluidflow Engineer**, an advanced fluid mechanics calculation tool.
        
        - **Sidebar Controls**: Select unit system (**SI** or **US Customary**), fluid properties, pipe geometry, surface roughness, and flow rate or velocity.
        - **Core Physics Engine**: All internal calculations strictly maintain SI dimensional consistency and execute standard fluid mechanics equations (Darcy-Weisbach, Colebrook-White, Swamee-Jain, Reynolds Number).
        - **Results Table**: Comprehensive state properties and method citations displayed via Pandas DataFrame with direct CSV download.
        - **Plotly Visualizations**: Dynamic Pressure Drop vs Flow Rate curve, Moody Diagram, and Head Loss charts.
        - **Engineering Interpretation & Diagnostics**: Automated physical evaluation of flow regimes, boundary layers, and warning flags for atypical velocities or pressure losses.
        - **AI Engineering Assistant**: Ask Gemini questions regarding boundary layers, pressure drop optimization, or mathematical methods.
        """
    )

# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.markdown("### ⚙️ System Configuration")

# Unit System
unit_system = st.sidebar.radio(
    "Unit System",
    options=["SI", "US Customary"],
    index=0,
    help="Select the engineering unit system. Internal calculations strictly maintain SI dimensional consistency.",
)
is_si = unit_system == "SI"

st.sidebar.markdown("---")
st.sidebar.markdown("#### 1. Fluid Selection")

fluid_options = ["Water (20°C)", "Air (20°C, 1 atm)", "Engine Oil (SAE 30, 20°C)", "Custom Fluid"]
fluid_choice = st.sidebar.selectbox(
    "Fluid",
    options=fluid_options,
    index=0,
    help="Select a standard fluid preset or define custom thermophysical properties.",
)

# Set standard property defaults
if fluid_choice == "Water (20°C)":
    default_rho_si = 998.2
    default_mu_si = 0.001002
elif fluid_choice == "Air (20°C, 1 atm)":
    default_rho_si = 1.204
    default_mu_si = 1.825e-5
elif fluid_choice == "Engine Oil (SAE 30, 20°C)":
    default_rho_si = 888.0
    default_mu_si = 0.290
else:
    default_rho_si = 1000.0
    default_mu_si = 0.001

if is_si:
    default_rho = default_rho_si
    default_mu = default_mu_si
    rho_unit = "kg/m³"
    mu_unit = "Pa·s"
else:
    default_rho = UnitConverter.density_from_si(default_rho_si, "lbm/ft3")
    default_mu = UnitConverter.dynamic_viscosity_from_si(default_mu_si, "cP")
    rho_unit = "lbm/ft³"
    mu_unit = "cP"

if fluid_choice == "Custom Fluid":
    col_f1, col_f2 = st.sidebar.columns(2)
    with col_f1:
        user_density = st.number_input(
            f"Density ρ [{rho_unit}]",
            min_value=0.001,
            value=float(round(default_rho, 3)),
            format="%.4f",
            help="Fluid mass density.",
        )
    with col_f2:
        user_viscosity = st.number_input(
            f"Viscosity μ [{mu_unit}]",
            min_value=1e-7,
            value=float(default_mu),
            format="%.6f",
            help="Dynamic fluid viscosity.",
        )
else:
    st.sidebar.caption(f"ρ = **{default_rho:.2f} {rho_unit}** | μ = **{default_mu:.6f} {mu_unit}**")
    user_density = default_rho
    user_viscosity = default_mu

st.sidebar.markdown("---")
st.sidebar.markdown("#### 2. Pipe Geometry & Material")

if is_si:
    default_diam = 0.100  # 100 mm = 0.1 m
    default_len = 50.0    # 50 m
    diam_unit = "m"
    len_unit = "m"
else:
    default_diam = 3.937  # ~4 inches
    default_len = 164.04  # ~50 m in ft
    diam_unit = "in"
    len_unit = "ft"

col_g1, col_g2 = st.sidebar.columns(2)
with col_g1:
    user_diam = st.number_input(
        f"Inner Diameter D [{diam_unit}]",
        min_value=0.0001,
        value=float(default_diam),
        format="%.4f",
        help="Inside pipe diameter.",
    )
with col_g2:
    user_length = st.number_input(
        f"Length L [{len_unit}]",
        min_value=0.01,
        value=float(default_len),
        format="%.2f",
        help="Total straight pipe length.",
    )

roughness_presets_keys = list(PIPE_ROUGHNESS_PRESETS.keys()) + ["Custom Roughness"]
rough_choice = st.sidebar.selectbox(
    "Roughness Preset",
    options=roughness_presets_keys,
    index=0,
    help="Select standard pipe wall material roughness or enter custom asperity height.",
)

if is_si:
    rough_unit = "mm"
    if rough_choice in PIPE_ROUGHNESS_PRESETS:
        default_rough = PIPE_ROUGHNESS_PRESETS[rough_choice]["roughness_m"] * 1000.0
    else:
        default_rough = 0.045
else:
    rough_unit = "in"
    if rough_choice in PIPE_ROUGHNESS_PRESETS:
        default_rough = PIPE_ROUGHNESS_PRESETS[rough_choice]["roughness_m"] / 0.0254
    else:
        default_rough = 0.00177

user_roughness = st.sidebar.number_input(
    f"Absolute Roughness ε [{rough_unit}]",
    min_value=0.0,
    value=float(round(default_rough, 6)),
    format="%.6f",
    help="Average pipe wall surface peak-to-valley roughness height.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("#### 3. Flow Operating Conditions")

flow_mode = st.sidebar.radio(
    "Flow Input Mode",
    options=["Volumetric flow rate", "Velocity"],
    index=0,
    horizontal=True,
    help="Specify volumetric flow rate (Q) or mean flow velocity (v).",
)

if is_si:
    if flow_mode == "Volumetric flow rate":
        flow_prompt = "Flow Rate Q [m³/s]"
        default_flow_val = 0.015  # 15 L/s
        flow_format = "%.5f"
    else:
        flow_prompt = "Velocity v [m/s]"
        default_flow_val = 1.91
        flow_format = "%.3f"
else:
    if flow_mode == "Volumetric flow rate":
        flow_prompt = "Flow Rate Q [ft³/s]"
        default_flow_val = 0.53
        flow_format = "%.4f"
    else:
        flow_prompt = "Velocity v [ft/s]"
        default_flow_val = 6.27
        flow_format = "%.2f"

user_flow_val = st.sidebar.number_input(
    flow_prompt,
    min_value=0.000001,
    value=float(default_flow_val),
    format=flow_format,
    help="Operating flow rate or velocity.",
)

# -----------------------------------------------------------------------------
# 4. CONVERSION TO SI & CALCULATION ENGINE
# -----------------------------------------------------------------------------
if is_si:
    diameter_si = float(user_diam)
    length_si = float(user_length)
    roughness_si = float(user_roughness) / 1000.0  # mm -> m
    density_si = float(user_density)
    viscosity_si = float(user_viscosity)
    if flow_mode == "Volumetric flow rate":
        flow_rate_si = float(user_flow_val)
        calc_mode_key = "flow_rate"
    else:
        velocity_si = float(user_flow_val)
        calc_mode_key = "velocity"
else:
    diameter_si = UnitConverter.length_to_si(user_diam, "in")
    length_si = UnitConverter.length_to_si(user_length, "ft")
    roughness_si = UnitConverter.length_to_si(user_roughness, "in")
    density_si = UnitConverter.density_to_si(user_density, "lbm/ft3")
    viscosity_si = UnitConverter.dynamic_viscosity_to_si(user_viscosity, "cP")
    if flow_mode == "Volumetric flow rate":
        flow_rate_si = UnitConverter.volumetric_flow_to_si(user_flow_val, "ft3/s")
        calc_mode_key = "flow_rate"
    else:
        velocity_si = UnitConverter.velocity_to_si(user_flow_val, "ft/s")
        calc_mode_key = "velocity"

# Input Validation
is_valid, validation_errors = validate_fluid_flow_inputs(
    diameter_m=diameter_si,
    length_m=length_si,
    roughness_m=roughness_si,
    density_kg_m3=density_si,
    viscosity_pa_s=viscosity_si,
    flow_value=flow_rate_si if calc_mode_key == "flow_rate" else velocity_si,
    flow_mode=calc_mode_key,
)

if not is_valid:
    st.error("### ⚠️ Input Validation Errors")
    for err in validation_errors:
        st.warning(f"- {err}")
    st.stop()

# Core SI calculations
area_si = calculate_cross_sectional_area(diameter_si)

if calc_mode_key == "flow_rate":
    velocity_si = calculate_velocity_from_flow(flow_rate_si, diameter_si)
else:
    flow_rate_si = calculate_flow_from_velocity(velocity_si, diameter_si)

mass_flow_si = calculate_mass_flow_rate(density_si, flow_rate_si)
reynolds = calculate_reynolds_number(density_si, velocity_si, diameter_si, viscosity_si)
relative_roughness = roughness_si / diameter_si
flow_regime, regime_description = classify_flow_regime(reynolds)

# Friction Factors
swamee_jain_f = calculate_swamee_jain_friction_factor(reynolds, relative_roughness)
colebrook_f, colebrook_iters, colebrook_converged = solve_colebrook_white_friction_factor(
    reynolds, relative_roughness
)

# Determine Darcy Friction Factor
if flow_regime == "Laminar":
    darcy_f = 64.0 / max(reynolds, 1e-5)
    f_method_label = "Laminar exact: f = 64 / Re"
elif flow_regime == "Transitional":
    t = (reynolds - 2300.0) / 1700.0
    f_lam_bound = 64.0 / 2300.0
    f_turb_bound, _, _ = solve_colebrook_white_friction_factor(4000.0, relative_roughness)
    darcy_f = (1.0 - t) * f_lam_bound + t * f_turb_bound
    f_method_label = "Transitional interpolation (2300 ≤ Re < 4000)"
else:
    darcy_f = colebrook_f
    f_method_label = f"Colebrook-White Implicit Solution (Newton-Raphson, {colebrook_iters} iters)"

diff_abs_f = abs(colebrook_f - swamee_jain_f)
diff_pct_f = (diff_abs_f / colebrook_f * 100.0) if colebrook_f > 0 else 0.0

# Pressure drop, gradient, head loss, power
pressure_drop_pa = calculate_pressure_drop(darcy_f, length_si, diameter_si, density_si, velocity_si)
dp_per_length_pa_m = pressure_drop_pa / length_si
head_loss_m = calculate_head_loss(darcy_f, length_si, diameter_si, velocity_si)
dynamic_pressure_pa = 0.5 * density_si * (velocity_si ** 2)
pumping_power_w = pressure_drop_pa * flow_rate_si
pumping_power_kw = pumping_power_w / 1000.0
pumping_power_hp = pumping_power_w / 745.699872

# -----------------------------------------------------------------------------
# 5. RESULTS DASHBOARD & METRICS
# -----------------------------------------------------------------------------
st.markdown("### 📊 Primary Hydraulic Results")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Reynolds Number (Re)",
        value=f"{reynolds:,.0f}",
        delta=f"Regime: {flow_regime}",
        delta_color="normal" if flow_regime == "Turbulent" else "inverse",
    )
    st.caption(f"Relative Roughness $\\epsilon/D$ = **{relative_roughness:.6f}**")

with col2:
    if is_si:
        vel_display = f"{velocity_si:.3f} m/s"
        area_display = f"{area_si:.5f} m²"
    else:
        vel_display = f"{UnitConverter.velocity_from_si(velocity_si, 'ft/s'):.2f} ft/s"
        area_display = f"{UnitConverter.area_from_si(area_si, 'ft2'):.4f} ft²"
    st.metric(
        label="Mean Velocity (v)",
        value=vel_display,
        delta=f"Area: {area_display}",
    )
    st.caption(f"Mass Flow ṁ = **{mass_flow_si:.3f} kg/s**")

with col3:
    if is_si:
        dp_display = f"{pressure_drop_pa:,.1f} Pa"
        dp_sub = f"{pressure_drop_pa / 1000.0:.2f} kPa"
    else:
        dp_psi = UnitConverter.pressure_from_si(pressure_drop_pa, "psi")
        dp_display = f"{dp_psi:.2f} psi"
        dp_sub = f"{dp_psi * 2.30665:.2f} ft H2O"
    st.metric(
        label="Frictional Pressure Drop (ΔP)",
        value=dp_display,
        delta=dp_sub,
        delta_color="off",
    )
    if is_si:
        st.caption(f"Grad $\\Delta P/L$ = **{dp_per_length_pa_m:.2f} Pa/m**")
    else:
        dp_l_imp = UnitConverter.pressure_from_si(pressure_drop_pa, "psi") / user_length
        st.caption(f"Grad $\\Delta P/L$ = **{dp_l_imp:.4f} psi/ft**")

with col4:
    if is_si:
        hf_display = f"{head_loss_m:.3f} m"
    else:
        hf_display = f"{UnitConverter.length_from_si(head_loss_m, 'ft'):.2f} ft"
    st.metric(
        label="Frictional Head Loss (h_f)",
        value=hf_display,
        delta=f"{pumping_power_kw:.3f} kW",
    )
    st.caption(f"Pumping Power = **{pumping_power_hp:.2f} hp**")

# Regime Physical State
if flow_regime == "Laminar":
    st.success(f"**Laminar Flow Regime (Re = {reynolds:,.1f} < 2300)**: {regime_description}")
elif flow_regime == "Transitional":
    st.warning(f"**Transitional Flow Regime (2300 ≤ Re = {reynolds:,.1f} < 4000)**: {regime_description}")
else:
    st.info(f"**Turbulent Flow Regime (Re = {reynolds:,.1f} ≥ 4000)**: {regime_description}")

# Friction Factor Details
st.markdown("#### 🔬 Friction Factor & Method Comparison")
f_col1, f_col2, f_col3, f_col4 = st.columns(4)

with f_col1:
    st.markdown("**Darcy Friction Factor (f)**")
    st.markdown(f"### `{darcy_f:.6f}`")
    st.caption(f"Method: {f_method_label}")

with f_col2:
    st.markdown("**Colebrook-White (Implicit)**")
    st.markdown(f"### `{colebrook_f:.6f}`")
    st.caption(f"Status: {colebrook_iters} iters (Newton-Raphson)")

with f_col3:
    st.markdown("**Swamee-Jain (Explicit)**")
    st.markdown(f"### `{swamee_jain_f:.6f}`")
    st.caption("Analytical direct equation")

with f_col4:
    st.markdown("**Difference (Colebrook vs SJ)**")
    st.markdown(f"### `{diff_pct_f:.3f}%`")
    st.caption(f"Absolute $\\Delta f$ = {diff_abs_f:.7f}")

# -----------------------------------------------------------------------------
# 6. RESULTS TABLE (PANDAS DATAFRAME & CSV EXPORT)
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📋 Complete Calculated State Table (Pandas Schema)")

table_rows = [
    {"Parameter": "Fluid Name", "Value": str(fluid_choice), "Unit": "-", "Method": "Thermophysical Database"},
    {"Parameter": "Density (ρ)", "Value": f"{density_si:.2f}", "Unit": "kg/m³", "Method": "Input / Property Library"},
    {"Parameter": "Dynamic Viscosity (μ)", "Value": f"{viscosity_si:.6e}", "Unit": "Pa·s", "Method": "Input / Property Library"},
    {"Parameter": "Kinematic Viscosity (ν)", "Value": f"{(viscosity_si / density_si):.6e}", "Unit": "m²/s", "Method": "ν = μ / ρ"},
    {"Parameter": "Pipe Inner Diameter (D)", "Value": f"{diameter_si:.4f}", "Unit": "m", "Method": "Conduit Geometry"},
    {"Parameter": "Pipe Length (L)", "Value": f"{length_si:.2f}", "Unit": "m", "Method": "Conduit Geometry"},
    {"Parameter": "Cross-Sectional Area (A)", "Value": f"{area_si:.6f}", "Unit": "m²", "Method": "A = π·D² / 4"},
    {"Parameter": "Absolute Roughness (ε)", "Value": f"{roughness_si*1000:.4f}", "Unit": "mm", "Method": "Material Wall Asperity"},
    {"Parameter": "Relative Roughness (ε/D)", "Value": f"{relative_roughness:.6f}", "Unit": "-", "Method": "ε / D"},
    {"Parameter": "Mean Flow Velocity (v)", "Value": f"{velocity_si:.4f}", "Unit": "m/s", "Method": "v = Q / A"},
    {"Parameter": "Volumetric Flow Rate (Q)", "Value": f"{flow_rate_si:.6f}", "Unit": "m³/s", "Method": "Q = v · A"},
    {"Parameter": "Mass Flow Rate (ṁ)", "Value": f"{mass_flow_si:.4f}", "Unit": "kg/s", "Method": "ṁ = ρ · Q"},
    {"Parameter": "Reynolds Number (Re)", "Value": f"{reynolds:,.1f}", "Unit": "-", "Method": "Re = (ρ · v · D) / μ"},
    {"Parameter": "Flow Regime", "Value": str(flow_regime), "Unit": "-", "Method": "Moody Re Boundary (<2300, 2300-4000, >4000)"},
    {"Parameter": "Darcy Friction Factor (f)", "Value": f"{darcy_f:.6f}", "Unit": "-", "Method": f_method_label},
    {"Parameter": "Colebrook-White f", "Value": f"{colebrook_f:.6f}", "Unit": "-", "Method": f"Newton-Raphson Implicit ({colebrook_iters} iters)"},
    {"Parameter": "Swamee-Jain f", "Value": f"{swamee_jain_f:.6f}", "Unit": "-", "Method": "Swamee-Jain Explicit Approximation"},
    {"Parameter": "Friction Factor Difference", "Value": f"{diff_pct_f:.4f}", "Unit": "%", "Method": "|f_CW - f_SJ| / f_CW · 100"},
    {"Parameter": "Frictional Pressure Drop (ΔP)", "Value": f"{pressure_drop_pa:,.2f}", "Unit": "Pa", "Method": "Darcy-Weisbach: ΔP = f · (L/D) · (ρ·v²/2)"},
    {"Parameter": "Pressure Gradient (ΔP/L)", "Value": f"{dp_per_length_pa_m:,.2f}", "Unit": "Pa/m", "Method": "ΔP / L"},
    {"Parameter": "Frictional Head Loss (h_f)", "Value": f"{head_loss_m:.4f}", "Unit": "m of fluid", "Method": "h_f = f · (L/D) · (v² / 2g)"},
    {"Parameter": "Dynamic Pressure (q)", "Value": f"{dynamic_pressure_pa:,.2f}", "Unit": "Pa", "Method": "q = 0.5 · ρ · v²"},
    {"Parameter": "Hydraulic Pumping Power", "Value": f"{pumping_power_kw:.4f}", "Unit": "kW", "Method": "P = ΔP · Q"},
    {"Parameter": "Brake Horsepower Equivalent", "Value": f"{pumping_power_hp:.3f}", "Unit": "hp", "Method": "P_kW / 0.7457"},
]

df_results = pd.DataFrame(table_rows)
st.dataframe(df_results, use_container_width=True, hide_index=True)

csv_bytes = df_results.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Results Table (CSV)",
    data=csv_bytes,
    file_name=f"gifty_fluidflow_results_{unit_system}.csv",
    mime="text/csv",
    help="Export all calculated parameters and methods to a CSV spreadsheet.",
)

# -----------------------------------------------------------------------------
# 7. INTERACTIVE CHARTS (PLOTLY)
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📈 Interactive Engineering Visualizations")

chart_tabs = st.tabs(["Pressure Drop vs Flow Rate", "Moody Diagram (f vs Re)", "Head Loss vs Flow Rate"])

with chart_tabs[0]:
    flow_unit_label = "m³/s" if is_si else "ft³/s"
    press_unit_label = "Pa" if is_si else "psi"
    q_disp_scale = 1.0 if is_si else UnitConverter.FLOW_CONVERSIONS["m3/s_to_ft3/s"]
    dp_disp_scale = 1.0 if is_si else UnitConverter.PRESSURE_CONVERSIONS["pa_to_psi"]
    
    fig_dp = plot_pressure_drop_vs_flow_rate(
        current_q_m3_s=flow_rate_si,
        current_dp_pa=pressure_drop_pa,
        diameter_m=diameter_si,
        length_m=length_si,
        density_kg_m3=density_si,
        viscosity_pa_s=viscosity_si,
        roughness_m=roughness_si,
        unit_system=unit_system,
        flow_unit=flow_unit_label,
        pressure_unit=press_unit_label,
        q_scale=q_disp_scale,
        dp_scale=dp_disp_scale,
    )
    st.plotly_chart(fig_dp, use_container_width=True)

with chart_tabs[1]:
    fig_moody = plot_moody_diagram(
        current_re=reynolds,
        current_f=darcy_f,
        current_rel_roughness=relative_roughness,
    )
    st.plotly_chart(fig_moody, use_container_width=True)

with chart_tabs[2]:
    head_unit_label = "m" if is_si else "ft"
    hf_disp_scale = 1.0 if is_si else UnitConverter.LENGTH_CONVERSIONS["m_to_ft"]
    
    fig_hf = plot_head_loss_vs_flow_rate(
        current_q_m3_s=flow_rate_si,
        current_hf_m=head_loss_m,
        diameter_m=diameter_si,
        length_m=length_si,
        density_kg_m3=density_si,
        viscosity_pa_s=viscosity_si,
        roughness_m=roughness_si,
        unit_system=unit_system,
        flow_unit=flow_unit_label,
        head_unit=head_unit_label,
        q_scale=q_disp_scale,
        hf_scale=hf_disp_scale,
    )
    st.plotly_chart(fig_hf, use_container_width=True)

# -----------------------------------------------------------------------------
# 8. ENGINEERING INTERPRETATION
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🧠 Engineering Interpretation")

col_ei1, col_ei2 = st.columns(2)

with col_ei1:
    st.markdown("#### Flow Regime & Reynolds Dynamics")
    if flow_regime == "Laminar":
        st.markdown(
            f"""
            - **Regime**: The flow is strictly **Laminar** ($Re = {reynolds:,.1f} < 2300$).
            - **Physical Mechanism**: Viscous shearing forces dominate inertial disturbances. The fluid particles travel in concentric, parallel streamlines with a classic parabolic velocity profile ($v_{{max}} = 2 v_{{avg}}$).
            - **Friction Factor Independence**: In laminar flow, surface roughness $\\epsilon$ has **no effect** on friction factor; $f = 64 / Re$.
            """
        )
    elif flow_regime == "Transitional":
        st.markdown(
            f"""
            - **Regime**: The flow is in the **Transitional Zone** ($2300 \\le Re = {reynolds:,.1f} < 4000$).
            - **Physical Mechanism**: The hydrodynamic boundary layer is unstable and intermittently oscillates between laminar viscous damping and turbulent eddy bursts.
            - **Design Advisory**: Flow rates in this zone can cause flow-induced vibrations and acoustic pulsing. If possible, resize diameter to shift operation decisively into laminar or turbulent regimes.
            """
        )
    else:
        st.markdown(
            f"""
            - **Regime**: The flow is fully **Turbulent** ($Re = {reynolds:,.1f} \\ge 4000$).
            - **Physical Mechanism**: Inertial momentum transfer overwhelms viscous damping. Intense turbulent eddies and cross-stream mixing create a flattened velocity profile with high wall shear stress.
            - **Roughness Influence**: At relative roughness $\\epsilon/D = {relative_roughness:.6f}$, pipe wall asperities penetrate the laminar viscous sublayer, directly dictating pressure dissipation.
            """
        )

with col_ei2:
    st.markdown("#### Pressure Drop & Correlation Deviation")
    st.markdown(
        f"""
        - **Pressure Loss Magnitude**: Total frictional pressure drop is **{pressure_drop_pa:,.1f} Pa** ({pressure_drop_pa/1000:.2f} kPa) across **{length_si:.1f} m** of conduit ($\Delta P/L = {dp_per_length_pa_m:.2f}$ Pa/m).
        - **Diameter Sensitivity**: Pressure drop scales with inverse diameter to the fifth power ($\Delta P \propto 1/D^5$) for a given flow rate. Increasing diameter by only 15% reduces pressure loss by ~50%.
        - **Colebrook-White vs Swamee-Jain**:
          - Colebrook-White: $f = {colebrook_f:.6f}$ (Solved implicitly in {colebrook_iters} iterations).
          - Swamee-Jain: $f = {swamee_jain_f:.6f}$ (Deviation: **{diff_pct_f:.3f}%**).
          - Analytical agreement is excellent ($\le 1.5\%$ deviation).
        """
    )

# -----------------------------------------------------------------------------
# 9. UNUSUAL PRESSURE LOSS & DIAGNOSTICS
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### ⚠️ Diagnostic Loss Analysis & Sizing Warnings")

diagnostic_messages = []

# Velocity checks
if velocity_si > 4.0:
    diagnostic_messages.append(
        f"**High Fluid Velocity ({velocity_si:.2f} m/s)**: Industrial liquid piping is typically sized for 1.5 - 3.0 m/s. Velocities exceeding 4 m/s increase acoustic noise, risk of water hammer surges, and accelerated wall erosion."
    )
elif velocity_si < 0.3:
    diagnostic_messages.append(
        f"**Low Fluid Velocity ({velocity_si:.2f} m/s)**: Low fluid velocities minimize frictional loss, but in process lines carrying suspended solids or slurries, velocities under 0.5 m/s may allow particle sedimentation."
    )

# Pressure Gradient checks
if dp_per_length_pa_m > 1000.0:
    diagnostic_messages.append(
        f"**Elevated Pressure Gradient ({dp_per_length_pa_m:.1f} Pa/m)**: System experiences high frictional resistance per unit length. Possible contributing factors: small pipe diameter ($D={diameter_si*1000:.1f}$ mm), high roughness, high fluid viscosity, or excessive flow rate."
    )

# Relative Roughness checks
if relative_roughness > 0.01:
    diagnostic_messages.append(
        f"**Substantial Pipe Roughness ($\epsilon/D = {relative_roughness:.5f}$)**: Wall roughness is high. The flow is in the complete turbulence regime where the friction factor becomes nearly independent of Reynolds number."
    )

# Geometric L/D ratio
ld_ratio = length_si / diameter_si
if ld_ratio < 30.0:
    diagnostic_messages.append(
        f"**Short Pipe Length ($L/D = {ld_ratio:.1f} < 30$)**: Hydrodynamic entrance length effects may contribute additional minor entrance loss not captured in the fully developed Darcy-Weisbach friction model."
    )

if diagnostic_messages:
    for msg in diagnostic_messages:
        st.warning(msg)
else:
    st.success(
        "✅ **Optimal Operating Conditions**: Fluid velocity, geometric aspect ratio ($L/D$), and pressure gradient are well-balanced within standard engineering design guidelines."
    )

# -----------------------------------------------------------------------------
# 10. AI ENGINEERING ASSISTANT (GEMINI)
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🤖 AI Engineering Assistant (Powered by Google Gemini)")

st.caption(
    "Ask engineering questions regarding your specific calculation results, boundary layers, Colebrook iterations, or system optimization."
)

gemini_key = get_gemini_api_key(getattr(st, "secrets", None))

col_ai_left, col_ai_right = st.columns([3, 1])

with col_ai_left:
    user_ai_query = st.text_input(
        "Engineering Query:",
        placeholder="e.g. Why is my calculated pressure loss so sensitive to pipe diameter?",
        key="ai_query_input",
    )

with col_ai_right:
    st.write("")
    st.write("")
    ask_ai_clicked = st.button("✨ Ask Gemini Assistant", use_container_width=True)

st.caption("Or choose a curated technical query:")
qp_cols = st.columns(4)
selected_qp = None

if qp_cols[0].button("🌊 Boundary Layer Physics", use_container_width=True):
    selected_qp = "Explain the physical meaning of my calculated Reynolds number and flow regime. Discuss the boundary layer and velocity profile."
if qp_cols[1].button("📉 Loss Optimization", use_container_width=True):
    selected_qp = "Analyze my calculated frictional pressure drop and pumping power. What specific engineering changes could reduce pressure loss?"
if qp_cols[2].button("🔄 Colebrook vs Swamee-Jain", use_container_width=True):
    selected_qp = "Compare the Colebrook-White and Swamee-Jain friction factors for this specific pipe configuration. Why is Colebrook implicit?"
if qp_cols[3].button("📏 20% Diameter Sensitivity", use_container_width=True):
    selected_qp = "What physical and hydraulic changes occur if we increase or decrease the pipe diameter by 20% at this same volumetric flow rate?"

active_prompt = selected_qp or (user_ai_query if ask_ai_clicked else None)

if active_prompt:
    calculation_summary = {
        "fluid": str(fluid_choice),
        "density_kg_m3": density_si,
        "dynamic_viscosity_pa_s": viscosity_si,
        "pipe_diameter_m": diameter_si,
        "pipe_length_m": length_si,
        "absolute_roughness_m": roughness_si,
        "relative_roughness": relative_roughness,
        "volumetric_flow_m3_s": flow_rate_si,
        "mean_velocity_m_s": velocity_si,
        "reynolds_number": reynolds,
        "flow_regime": flow_regime,
        "darcy_friction_factor": darcy_f,
        "colebrook_friction_factor": colebrook_f,
        "swamee_jain_friction_factor": swamee_jain_f,
        "diff_pct_f": diff_pct_f,
        "pressure_drop_pa": pressure_drop_pa,
        "pressure_gradient_pa_m": dp_per_length_pa_m,
        "head_loss_m": head_loss_m,
        "pumping_power_kw": pumping_power_kw,
    }

    with st.spinner("Analyzing fluid mechanics physics with Gemini AI..."):
        ai_result = generate_engineering_explanation(
            user_query=active_prompt,
            calculation_summary=calculation_summary,
            api_key=gemini_key,
        )

    if ai_result["success"]:
        st.markdown("#### 💡 AI Engineering Response:")
        st.markdown(ai_result["text"])
    else:
        st.info(ai_result["text"])

# -----------------------------------------------------------------------------
# 11. GOVERNING EQUATIONS & ASSUMPTIONS
# -----------------------------------------------------------------------------
st.markdown("---")

with st.expander("📐 Governing Equations & Mathematical Formulations", expanded=False):
    st.markdown(
        r"""
        #### 1. Continuity & Flow Velocity
        $$\text{Area: } A = \frac{\pi D^2}{4}, \quad \text{Mean Velocity: } v = \frac{Q}{A}, \quad \text{Mass Flow: } \dot{m} = \rho Q$$

        #### 2. Reynolds Number
        $$Re = \frac{\rho v D}{\mu} = \frac{v D}{\nu}$$
        - **Laminar**: $Re < 2300$
        - **Transitional**: $2300 \le Re < 4000$
        - **Turbulent**: $Re \ge 4000$

        #### 3. Darcy-Weisbach Equation for Frictional Pressure Drop
        $$\Delta P = f \cdot \frac{L}{D} \cdot \left(\frac{\rho v^2}{2}\right), \quad h_f = f \cdot \frac{L}{D} \cdot \left(\frac{v^2}{2g}\right)$$

        #### 4. Friction Factor Formulations
        - **Laminar Exact Formula**:
          $$f = \frac{64}{Re}$$
        - **Colebrook-White Implicit Equation** (Turbulent):
          $$\frac{1}{\sqrt{f}} = -2.0 \log_{10} \left( \frac{\epsilon}{3.7 D} + \frac{2.51}{Re \sqrt{f}} \right)$$
        - **Swamee-Jain Explicit Approximation**:
          $$f = \frac{0.25}{\left[ \log_{10} \left( \frac{\epsilon}{3.7 D} + \frac{5.74}{Re^{0.9}} \right) \right]^2}$$
        """
    )

with st.expander("📋 Assumptions & Engineering Limitations", expanded=False):
    st.markdown(
        """
        - **Incompressible Fluid Flow**: Assumes constant density ($\rho$) throughout conduit length (valid for all liquids and gases at Mach number $M < 0.3$).
        - **Fully Developed Steady-State Flow**: Hydrodynamic entrance length effects are not added to macro friction factor calculations.
        - **Newtonian Behavior**: Dynamic viscosity ($\mu$) is independent of shear rate.
        - **Conduit Geometry**: Assumes circular pipe cross-section of uniform diameter and constant wall roughness.
        - **Standard Gravity**: Standard acceleration of gravity $g = 9.80665\text{ m/s}^2$ is utilized.
        - **Design Verification**: Calculations serve preliminary hydraulic engineering sizing. Final safety design must comply with ASME B31.3 / B31.1 piping codes.
        """
    )

st.markdown(
    """
    <div style='text-align: center; color: #64748b; font-size: 0.8rem; padding: 20px 0;'>
        <b>Gifty Fluidflow Engineer</b> &copy; 2026 | General Fluid Mechanics & Fluid Flow Engineering Calculator<br>
        Streamlit &bull; NumPy &bull; Pandas &bull; Plotly &bull; Google Gemini
    </div>
    """,
    unsafe_allow_html=True,
)
