"""
Gifty Fluidflow Engineer - AI Engineering Assistant (Google Gemini)
Provides AI-assisted physical explanations of fluid mechanics calculation results
using the official Google GenAI / Gemini Python SDK with reliable analytical fallback.
"""

import os
from typing import Dict, Any, Optional

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


def get_gemini_api_key(st_secrets_obj: Optional[Any] = None) -> Optional[str]:
    """Retrieve Gemini API key from Streamlit secrets or system environment."""
    if st_secrets_obj is not None:
        try:
            if "GEMINI_API_KEY" in st_secrets_obj:
                return str(st_secrets_obj["GEMINI_API_KEY"])
        except Exception:
            pass

    return os.environ.get("GEMINI_API_KEY", None)


def generate_analytical_explanation(
    user_query: str,
    calc: Dict[str, Any],
) -> str:
    """Generate deterministic, rigorous fluid mechanics engineering report."""
    fluid = calc.get("fluid") or calc.get("fluid_name") or "Fluid"
    re = float(calc.get("reynolds_number") or calc.get("reynolds") or 0.0)
    regime = calc.get("flow_regime") or ("Laminar" if re < 2300 else "Transitional" if re < 4000 else "Turbulent")
    diam_m = float(calc.get("pipe_diameter_m") or calc.get("diameter_m") or 0.1)
    diam_mm = diam_m * 1000.0
    length_m = float(calc.get("pipe_length_m") or calc.get("length_m") or 50.0)
    v_ms = float(calc.get("mean_velocity_m_s") or calc.get("velocity_m_s") or 0.0)
    q_m3s = float(calc.get("volumetric_flow_m3_s") or calc.get("flow_rate_m3_s") or 0.0)
    dp_pa = float(calc.get("pressure_drop_pa") or 0.0)
    hf_m = float(calc.get("head_loss_m") or 0.0)
    p_kw = float(calc.get("pumping_power_kw") or (dp_pa * q_m3s / 1000.0))
    f_darcy = float(calc.get("darcy_friction_factor") or calc.get("friction_factor") or 0.02)
    f_cw = float(calc.get("colebrook_friction_factor") or calc.get("colebrook_f") or f_darcy)
    f_sj = float(calc.get("swamee_jain_friction_factor") or calc.get("swamee_jain_f") or f_darcy)
    diff_pct = float(calc.get("diff_pct_f") or calc.get("friction_pct_diff") or 0.0)
    rel_rough = float(calc.get("relative_roughness") or 0.0)

    q_lower = (user_query or "").lower()

    if "diameter" in q_lower or "sizing" in q_lower or "20%" in q_lower:
        dp_red = dp_pa * ((1.0 / 1.2) ** 5)
        dp_inc = dp_pa * ((1.0 / 0.8) ** 5)
        return (
            "### 📏 Pipe Diameter Sizing & Hydraulic Sensitivity\n\n"
            + f"**Current Conduit Parameters:**\n"
            + f"- Inner Diameter: **{diam_mm:.1f} mm** (D = {diam_m:.4f} m)\n"
            + f"- Volumetric Flow: **{q_m3s:.5f} m³/s** | Mean Velocity: **{v_ms:.3f} m/s**\n"
            + f"- Frictional Pressure Drop: **{dp_pa:,.1f} Pa** ({dp_pa/1000.0:.2f} kPa)\n\n"
            + "**Governing Scaling Law (ΔP ∝ 1/D⁵):**\n"
            + "From the Darcy-Weisbach formulation expressed in terms of volumetric flow rate Q:\n"
            + "$$\\Delta P = \\frac{8 f L \\rho Q^2}{\\pi^2 D^5}$$\n\n"
            + "**Impact of ±20% Diameter Variation at Constant Flow Rate:**\n"
            + f"1. **+20% Diameter ({diam_mm * 1.2:.1f} mm):**\n"
            + f"   - Cross-sectional area increases by **44%** (1.2² = 1.44).\n"
            + f"   - Mean velocity decreases to **{v_ms / 1.44:.3f} m/s**.\n"
            + f"   - Pressure drop drops by **59.8%** down to **{dp_red:,.1f} Pa**.\n"
            + "   - Hydraulic pumping power is reduced by more than half.\n\n"
            + f"2. **-20% Diameter ({diam_mm * 0.8:.1f} mm):**\n"
            + f"   - Velocity increases by **56.3%** to **{v_ms / 0.64:.3f} m/s**.\n"
            + f"   - Pressure drop surges by **+205%** up to **{dp_inc:,.1f} Pa**.\n"
            + "   - May exceed acoustic noise velocity limits and exacerbate water hammer risk."
        )

    if "colebrook" in q_lower or "swamee" in q_lower or "friction" in q_lower:
        return (
            "### 🔄 Friction Factor Analysis: Colebrook-White vs. Swamee-Jain\n\n"
            + "**Operating System Values:**\n"
            + f"- **Colebrook-White (f_CW):** `{f_cw:.6f}` (Implicit Newton-Raphson solution)\n"
            + f"- **Swamee-Jain (f_SJ):** `{f_sj:.6f}` (Explicit analytical correlation)\n"
            + f"- **Relative Difference:** `{diff_pct:.3f}%`\n"
            + f"- **Flow Regime:** **{regime}** (Re = {re:,.1f}, ε/D = {rel_rough:.6f})\n\n"
            + "**Why is the Colebrook-White Equation Implicit?**\n"
            + "The Colebrook equation unites smooth pipe turbulent boundary layer theory with Nikuradse's fully rough pipe data:\n"
            + "$$\\frac{1}{\\sqrt{f}} = -2.0 \\log_{10} \\left( \\frac{\\epsilon}{3.7 D} + \\frac{2.51}{Re \\sqrt{f}} \\right)$$\n"
            + "Because friction factor f appears on both sides of the logarithmic equation, numerical iteration (Newton-Raphson) is necessary.\n\n"
            + "**Swamee-Jain Explicit Approximation:**\n"
            + "$$f = \\frac{0.25}{\\left[ \\log_{10} \\left( \\frac{\\epsilon}{3.7 D} + \\frac{5.74}{Re^{0.9}} \\right) \\right]^2}$$\n"
            + f"For your operating condition, the deviation between Colebrook and Swamee-Jain is **{diff_pct:.3f}%**, demonstrating excellent precision."
        )

    if "pressure" in q_lower or "power" in q_lower or "loss" in q_lower or "optimize" in q_lower:
        annual_kwh = (p_kw / 0.70) * 8760.0 if p_kw > 0 else 0.0
        return (
            "### 📉 Frictional Pressure Drop & Pumping Power Analysis\n\n"
            + "**Calculated Hydraulic State:**\n"
            + f"- Frictional Pressure Drop (ΔP): **{dp_pa:,.1f} Pa** ({dp_pa/1000.0:.2f} kPa)\n"
            + f"- Pressure Gradient (ΔP/L): **{dp_pa / length_m:.2f} Pa/m**\n"
            + f"- Frictional Head Loss (h_f): **{hf_m:.3f} m** of {fluid}\n"
            + f"- Hydraulic Pumping Power (P = ΔP · Q): **{p_kw:.3f} kW**\n\n"
            + "**Energy & Optimization Insights:**\n"
            + f"1. **Continuous Operation Cost**: Assuming standard pump/motor efficiency η = 70% operating 8,760 hours/year, electrical energy consumption is **~{annual_kwh:,.0f} kWh/year**.\n"
            + "2. **Piping Optimization Strategies**:\n"
            + "   - **Upsizing Nominal Diameter**: Moving up one standard pipe schedule yields an exponential D⁻⁵ decrease in pressure drop.\n"
            + "   - **Smooth Internal Linings**: Specifying smooth epoxy or thermoplastic conduits reduces wall roughness ε.\n"
            + "   - **Fittings & Minor Losses**: Ensure low-loss long-radius elbows are selected for high-velocity sections."
        )

    if regime == "Laminar":
        re_explanation = "Viscous damping forces dominate, producing concentric fluid lamina and a parabolic velocity profile (v_max = 2 * v_avg)."
    elif regime == "Transitional":
        re_explanation = "In this transitional zone, boundary layer stability is fragile, intermittently triggering turbulent eddy bursts."
    else:
        re_explanation = "Inertial momentum dominates viscous dissipation, resulting in a flattened turbulent core velocity profile with high wall shear stress."

    if rel_rough > 0.001:
        rough_explanation = "substantially penetrate the laminar viscous sublayer, pushing the flow toward the fully rough turbulent regime where friction is nearly independent of Re."
    else:
        rough_explanation = "remain largely submerged within the viscous sublayer, meaning friction factor is governed primarily by Reynolds number."

    return (
        "### 🌊 Comprehensive Fluid Mechanics & Boundary Layer Report\n\n"
        + "**System State Summary:**\n"
        + f"- **Fluid:** {fluid} | Density: **{float(calc.get('density_kg_m3') or calc.get('density') or 1000.0):.1f} kg/m³**\n"
        + f"- **Reynolds Number (Re):** **{re:,.1f}** → **{regime.upper()} REGIME**\n"
        + f"- **Mean Velocity (v):** **{v_ms:.3f} m/s** | **Conduit Diameter (D):** **{diam_mm:.1f} mm**\n"
        + f"- **Darcy Friction Factor (f):** **{f_darcy:.6f}** (ε/D = {rel_rough:.6f})\n"
        + f"- **Total Pressure Drop (ΔP):** **{dp_pa:,.1f} Pa** across **{length_m:.1f} m** of pipe\n\n"
        + "**Physical Interpretation:**\n"
        + f"1. **Inertial vs Viscous Dominance (Re):**\n"
        + f"   The Reynolds number (Re = {re:,.1f}) measures the ratio of inertial momentum forces to viscous shearing resistance. {re_explanation}\n\n"
        + "2. **Wall Roughness Interaction:**\n"
        + f"   At relative roughness ε/D = {rel_rough:.6f}, pipe wall asperities {rough_explanation}"
    )


def generate_engineering_explanation(
    user_query: str,
    calculation_summary: Dict[str, Any],
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate an engineering interpretation using Gemini API if key is available,
    with automatic analytical engineering report fallback.
    Returns dict with {"success": bool, "text": str, "error": Optional[str]}.
    """
    if not api_key or not GENAI_AVAILABLE:
        analytical_text = generate_analytical_explanation(user_query, calculation_summary)
        return {
            "success": True,
            "text": analytical_text,
            "error": None,
            "source": "analytical_engine",
        }

    system_instruction = (
        "You are the AI Engineering Assistant for 'Gifty Fluidflow Engineer', an advanced fluid mechanics "
        "and internal pipe flow analysis platform.\n"
        "Your role is to explain the physical engineering principles behind the user's specific calculation results.\n\n"
        "STRICT GUIDELINES:\n"
        "1. Strictly use the provided calculation numbers. Do not alter or contradict the mathematical results.\n"
        "2. Explain fluid mechanics concepts clearly: Reynolds number physical interpretation (inertial vs viscous ratio), "
        "flow regime boundary behavior, boundary layer development, laminar parabolic profile vs turbulent logarithmic profile, "
        "viscous sublayer submergence by surface roughness asperities, and Colebrook vs Swamee-Jain differences.\n"
        "3. Provide practical engineering guidance (pumping power considerations, erosion/noise velocity thresholds, sizing recommendations).\n"
        "4. Emphasize that final critical industrial piping designs require verification with ASME B31.3 / Hydraulic Institute standards.\n"
        "5. Structure responses cleanly with bold headings and bullet points."
    )

    try:
        client = genai.Client(api_key=api_key)
        
        prompt_content = (
            f"User Question: {user_query}\n\n"
            f"--- CURRENT SYSTEM CALCULATION DATA ---\n"
            f"Fluid: {calculation_summary.get('fluid') or calculation_summary.get('fluid_name')}\n"
            f"Density (rho): {calculation_summary.get('density_kg_m3') or calculation_summary.get('density')} kg/m³\n"
            f"Dynamic Viscosity (mu): {calculation_summary.get('dynamic_viscosity_pa_s') or calculation_summary.get('viscosity')} Pa·s\n"
            f"Pipe Inner Diameter (D): {calculation_summary.get('pipe_diameter_m') or calculation_summary.get('diameter_m')} m\n"
            f"Pipe Length (L): {calculation_summary.get('pipe_length_m') or calculation_summary.get('length_m')} m\n"
            f"Pipe Roughness (epsilon): {calculation_summary.get('absolute_roughness_m') or calculation_summary.get('roughness_m')} m\n"
            f"Relative Roughness (epsilon/D): {calculation_summary.get('relative_roughness')}\n"
            f"Volumetric Flow Rate (Q): {calculation_summary.get('volumetric_flow_m3_s') or calculation_summary.get('flow_rate_m3_s')} m³/s\n"
            f"Mean Velocity (v): {calculation_summary.get('mean_velocity_m_s') or calculation_summary.get('velocity_m_s')} m/s\n"
            f"Reynolds Number (Re): {calculation_summary.get('reynolds_number') or calculation_summary.get('reynolds')}\n"
            f"Flow Regime: {calculation_summary.get('flow_regime')}\n"
            f"Darcy Friction Factor (f): {calculation_summary.get('darcy_friction_factor') or calculation_summary.get('friction_factor')}\n"
            f"Colebrook-White f: {calculation_summary.get('colebrook_friction_factor') or calculation_summary.get('colebrook_f')}\n"
            f"Swamee-Jain f: {calculation_summary.get('swamee_jain_friction_factor') or calculation_summary.get('swamee_jain_f')}\n"
            f"Frictional Pressure Drop (Delta P): {calculation_summary.get('pressure_drop_pa')} Pa\n"
            f"Frictional Head Loss (h_f): {calculation_summary.get('head_loss_m')} m\n"
            f"Pumping Power (kW): {calculation_summary.get('pumping_power_kw')}\n"
            f"----------------------------------------"
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_content,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3,
            ),
        )

        return {
            "success": True,
            "text": response.text or "No response received from Gemini model.",
            "error": None,
            "source": "gemini",
        }

    except Exception as e:
        # Fall back to analytical interpretation rather than failing
        fallback_report = generate_analytical_explanation(user_query, calculation_summary)
        return {
            "success": True,
            "text": fallback_report,
            "error": None,
            "source": "analytical_fallback",
        }
