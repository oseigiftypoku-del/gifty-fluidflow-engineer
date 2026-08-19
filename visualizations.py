"""
Gifty Fluidflow Engineer - Interactive Visualizations Engine (Plotly)
Generates high-precision engineering charts for Pressure Drop vs Flow Rate,
the Moody Diagram (f vs Re with relative roughness curves), and Head Loss vs Flow Rate.
"""

import math
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, List
from calculations import (
    calculate_reynolds_number,
    calculate_darcy_friction_factor,
    calculate_pressure_drop,
    calculate_head_loss,
    calculate_velocity_from_flow,
    calculate_swamee_jain_friction_factor,
    solve_colebrook_white_friction_factor,
)


def plot_pressure_drop_vs_flow_rate(
    current_q_m3_s: float,
    current_dp_pa: float,
    diameter_m: float,
    length_m: float,
    density_kg_m3: float,
    viscosity_pa_s: float,
    roughness_m: float,
    unit_system: str = "SI",
    flow_unit: str = "m³/s",
    pressure_unit: str = "Pa",
    q_scale: float = 1.0,
    dp_scale: float = 1.0,
) -> go.Figure:
    """
    Generate interactive Pressure Drop vs Volumetric Flow Rate curve
    spanning across a 0.1x to 2.5x operating range with current point highlighted.
    """
    q_min = max(current_q_m3_s * 0.05, 1e-6)
    q_max = current_q_m3_s * 2.2
    
    q_array_m3_s = np.linspace(q_min, q_max, 120)
    dp_array = []
    v_array = []
    re_array = []
    regimes = []

    rel_roughness = roughness_m / diameter_m

    for q_val in q_array_m3_s:
        v_val = calculate_velocity_from_flow(q_val, diameter_m)
        re_val = calculate_reynolds_number(density_kg_m3, v_val, diameter_m, viscosity_pa_s)
        f_res = calculate_darcy_friction_factor(re_val, rel_roughness)
        dp_val = calculate_pressure_drop(f_res["friction_factor"], length_m, diameter_m, density_kg_m3, v_val)
        
        q_display = q_val * q_scale
        dp_display = dp_val * dp_scale
        v_array.append(v_val)
        re_array.append(re_val)
        dp_array.append(dp_display)
        
        if re_val < 2300:
            regimes.append("Laminar")
        elif re_val < 4000:
            regimes.append("Transitional")
        else:
            regimes.append("Turbulent")

    q_display_array = q_array_m3_s * q_scale
    current_q_display = current_q_m3_s * q_scale
    current_dp_display = current_dp_pa * dp_scale

    fig = go.Figure()

    # Base curve
    fig.add_trace(
        go.Scatter(
            x=q_display_array,
            y=dp_array,
            mode="lines",
            name="System Resistance Curve",
            line=dict(color="#0284c7", width=3.5),
            customdata=np.stack((v_array, re_array, regimes), axis=-1),
            hovertemplate=(
                f"<b>Flow Rate (Q):</b> %{{x:.4f}} {flow_unit}<br>"
                f"<b>Pressure Drop (ΔP):</b> %{{y:.2f}} {pressure_unit}<br>"
                "<b>Mean Velocity:</b> %{customdata[0]:.2f} m/s<br>"
                "<b>Reynolds (Re):</b> %{customdata[1]:,.0f}<br>"
                "<b>Flow Regime:</b> %{customdata[2]}<extra></extra>"
            ),
        )
    )

    # Current Operating Point Marker
    fig.add_trace(
        go.Scatter(
            x=[current_q_display],
            y=[current_dp_display],
            mode="markers+text",
            name="Current Operating Point",
            marker=dict(color="#ef4444", size=14, symbol="circle", line=dict(color="#ffffff", width=2)),
            text=[" Operating Point"],
            textposition="top left",
            textfont=dict(color="#0f172a", size=12, family="Inter, sans-serif"),
            hovertemplate=(
                f"<b>CURRENT OPERATING POINT</b><br>"
                f"Flow Rate: {current_q_display:.4f} {flow_unit}<br>"
                f"Pressure Drop: {current_dp_display:.2f} {pressure_unit}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"<b>System Pressure Drop vs Volumetric Flow Rate</b>",
            font=dict(size=16, color="#0f172a", family="Inter, sans-serif"),
        ),
        xaxis=dict(
            title=f"Volumetric Flow Rate Q [{flow_unit}]",
            gridcolor="#e2e8f0",
            zerolinecolor="#cbd5e1",
            showline=True,
            linecolor="#94a3b8",
        ),
        yaxis=dict(
            title=f"Frictional Pressure Drop ΔP [{pressure_unit}]",
            gridcolor="#e2e8f0",
            zerolinecolor="#cbd5e1",
            showline=True,
            linecolor="#94a3b8",
        ),
        template="plotly_white",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=40, t=70, b=50),
    )

    return fig


def plot_moody_friction_factor_chart(
    current_re: float,
    current_f: float,
    current_rel_roughness: float,
) -> go.Figure:
    """
    Generate classic engineering Moody Diagram (f vs Re) on log-log scale.
    Displays:
    - Laminar boundary line f = 64/Re (Re = 600 to 2300)
    - Shaded transitional uncertainty region (2300 <= Re < 4000)
    - Turbulent Colebrook curves across representative relative roughness values (epsilon/D)
    - Operating coordinate (Re_0, f_0) clearly indicated
    """
    fig = go.Figure()

    # 1. Laminar Line: Re from 500 to 2300
    re_lam = np.logspace(np.log10(500), np.log10(2300), 50)
    f_lam = 64.0 / re_lam
    fig.add_trace(
        go.Scatter(
            x=re_lam,
            y=f_lam,
            mode="lines",
            name="Laminar (f = 64/Re)",
            line=dict(color="#16a34a", width=2.5),
            hovertemplate="<b>Laminar Line</b><br>Re: %{x:,.0f}<br>f: %{y:.5f}<extra></extra>",
        )
    )

    # 2. Transitional Band Shading (2300 <= Re <= 4000)
    fig.add_vrect(
        x0=2300,
        x1=4000,
        fillcolor="#fef3c7",
        opacity=0.5,
        layer="below",
        line_width=0,
        annotation_text="Transitional Zone",
        annotation_position="top left",
        annotation_font=dict(size=10, color="#b45309"),
    )

    # 3. Turbulent Curves (Colebrook) for standard relative roughness values
    roughness_curves = [
        (0.05, "#dc2626", "ε/D = 0.05"),
        (0.02, "#ea580c", "ε/D = 0.02"),
        (0.01, "#d97706", "ε/D = 0.01"),
        (0.004, "#0284c7", "ε/D = 0.004"),
        (0.001, "#2563eb", "ε/D = 0.001"),
        (0.0001, "#4f46e5", "ε/D = 0.0001"),
        (0.00001, "#7c3aed", "ε/D = 0.00001"),
        (0.000001, "#64748b", "Smooth Pipe (ε/D ≈ 0)"),
    ]

    re_turb = np.logspace(np.log10(4000), np.log10(1e8), 70)

    for rr_val, color_hex, label in roughness_curves:
        f_curve = []
        for re_val in re_turb:
            f_val, _, _ = solve_colebrook_white_friction_factor(re_val, rr_val)
            f_curve.append(f_val)
        
        fig.add_trace(
            go.Scatter(
                x=re_turb,
                y=f_curve,
                mode="lines",
                name=label,
                line=dict(color=color_hex, width=1.4, dash="solid" if rr_val > 0.00001 else "dash"),
                hovertemplate=f"<b>{label}</b><br>Re: %{{x:,.0f}}<br>f: %{{y:.5f}}<extra></extra>",
            )
        )

    # 4. User System Roughness Curve (if distinct from presets)
    if current_rel_roughness > 0:
        user_f_curve = []
        for re_val in re_turb:
            f_val, _, _ = solve_colebrook_white_friction_factor(re_val, current_rel_roughness)
            user_f_curve.append(f_val)
        
        fig.add_trace(
            go.Scatter(
                x=re_turb,
                y=user_f_curve,
                mode="lines",
                name=f"Current Pipe (ε/D = {current_rel_roughness:.5f})",
                line=dict(color="#0284c7", width=3.0),
                hovertemplate="<b>Your System Pipe Curve</b><br>Re: %{x:,.0f}<br>f: %{y:.5f}<extra></extra>",
            )
        )

    # 5. Operating Point
    fig.add_trace(
        go.Scatter(
            x=[max(current_re, 500)],
            y=[current_f],
            mode="markers+text",
            name="Operating Point",
            marker=dict(color="#e11d48", size=14, symbol="diamond", line=dict(color="#ffffff", width=2)),
            text=[" Operating Point"],
            textposition="bottom right",
            textfont=dict(color="#0f172a", size=12, family="Inter, sans-serif"),
            hovertemplate=(
                f"<b>CURRENT SYSTEM POINT</b><br>"
                f"Reynolds Number (Re): {current_re:,.0f}<br>"
                f"Friction Factor (f): {current_f:.5f}<br>"
                f"Relative Roughness (ε/D): {current_rel_roughness:.5f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Moody Friction Factor Diagram (Darcy-Weisbach f vs Re)</b>",
            font=dict(size=16, color="#0f172a", family="Inter, sans-serif"),
        ),
        xaxis=dict(
            title="Reynolds Number (Re) [Logarithmic Scale]",
            type="log",
            range=[np.log10(500), np.log10(1e8)],
            gridcolor="#e2e8f0",
            zerolinecolor="#cbd5e1",
            showline=True,
            linecolor="#94a3b8",
        ),
        yaxis=dict(
            title="Darcy Friction Factor (f) [Logarithmic Scale]",
            type="log",
            range=[np.log10(0.007), np.log10(0.12)],
            gridcolor="#e2e8f0",
            zerolinecolor="#cbd5e1",
            showline=True,
            linecolor="#94a3b8",
        ),
        template="plotly_white",
        hovermode="closest",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#e2e8f0",
            borderwidth=1,
            font=dict(size=10),
        ),
        margin=dict(l=60, r=40, t=70, b=50),
    )

    return fig


def plot_head_loss_vs_flow_rate(
    current_q_m3_s: float,
    current_hf_m: float,
    diameter_m: float,
    length_m: float,
    density_kg_m3: float,
    viscosity_pa_s: float,
    roughness_m: float,
    head_unit: str = "m",
    flow_unit: str = "m³/s",
    head_scale: float = 1.0,
    q_scale: float = 1.0,
) -> go.Figure:
    """Generate Head Loss h_f [m or ft] vs Volumetric Flow Rate chart."""
    q_min = max(current_q_m3_s * 0.05, 1e-6)
    q_max = current_q_m3_s * 2.2
    
    q_array = np.linspace(q_min, q_max, 100)
    hf_array = []
    v_array = []

    rel_roughness = roughness_m / diameter_m

    for q_val in q_array:
        v_val = calculate_velocity_from_flow(q_val, diameter_m)
        re_val = calculate_reynolds_number(density_kg_m3, v_val, diameter_m, viscosity_pa_s)
        f_res = calculate_darcy_friction_factor(re_val, rel_roughness)
        hf_val = calculate_head_loss(f_res["friction_factor"], length_m, diameter_m, v_val)
        
        hf_array.append(hf_val * head_scale)
        v_array.append(v_val)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=q_array * q_scale,
            y=hf_array,
            mode="lines",
            name="Frictional Head Loss (h_f)",
            line=dict(color="#059669", width=3),
            customdata=v_array,
            hovertemplate=(
                f"Flow Rate: %{{x:.4f}} {flow_unit}<br>"
                f"Head Loss: %{{y:.2f}} {head_unit}<br>"
                "Velocity: %{customdata:.2f} m/s<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[current_q_m3_s * q_scale],
            y=[current_hf_m * head_scale],
            mode="markers+text",
            name="Operating Head Loss",
            marker=dict(color="#dc2626", size=13, symbol="circle"),
            text=[" Operating Point"],
            textposition="top left",
            hovertemplate=(
                f"<b>OPERATING HEAD LOSS</b><br>"
                f"Flow: {current_q_m3_s * q_scale:.4f} {flow_unit}<br>"
                f"Head Loss: {current_hf_m * head_scale:.2f} {head_unit}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Frictional Head Loss vs Flow Rate</b>",
            font=dict(size=16, color="#0f172a", family="Inter, sans-serif"),
        ),
        xaxis=dict(title=f"Volumetric Flow Rate Q [{flow_unit}]", gridcolor="#e2e8f0"),
        yaxis=dict(title=f"Head Loss h_f [{head_unit}]", gridcolor="#e2e8f0"),
        template="plotly_white",
        margin=dict(l=60, r=40, t=70, b=50),
    )

    return fig
