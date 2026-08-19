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
"""
AI Engineering Assistant for Gifty Fluidflow Engineer.

Provides Google Gemini-powered engineering explanations.
"""

import os
from typing import Any, Dict, Optional


def get_gemini_api_key(secrets: Optional[Any] = None) -> Optional[str]:
    """
    Retrieve the Gemini API key from Streamlit secrets or environment variables.

    Priority:
    1. Streamlit secrets
    2. GEMINI_API_KEY environment variable
    3. GOOGLE_API_KEY environment variable

    Returns:
        API key string if available, otherwise None.
    """

    # Try Streamlit secrets
    if secrets is not None:
        try:
            if "GEMINI_API_KEY" in secrets:
                return str(secrets["GEMINI_API_KEY"])

            if "GOOGLE_API_KEY" in secrets:
                return str(secrets["GOOGLE_API_KEY"])
        except Exception:
            pass

    # Try environment variables
    api_key = os.getenv("GEMINI_API_KEY")

    if api_key:
        return api_key

    api_key = os.getenv("GOOGLE_API_KEY")

    if api_key:
        return api_key

    return None


def generate_engineering_explanation(
    user_query: str,
    calculation_summary: Dict[str, Any],
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate an engineering explanation using Google Gemini.

    Args:
        user_query:
            Engineering question from the user.

        calculation_summary:
            Dictionary containing the current hydraulic calculation results.

        api_key:
            Google Gemini API key.

    Returns:
        Dictionary containing:
            success: Boolean
            text: Response text
    """

    if not user_query or not user_query.strip():
        return {
            "success": False,
            "text": "Please enter an engineering question."
        }

    if not api_key:
        return {
            "success": False,
            "text": (
                "Gemini API key not configured. "
                "Add GEMINI_API_KEY to Streamlit secrets "
                "to enable the AI Engineering Assistant."
            )
        }

    try:
        from google import genai

        client = genai.Client(api_key=api_key)

        engineering_context = "\n".join(
            f"- {key}: {value}"
            for key, value in calculation_summary.items()
        )

        prompt = f"""
You are a professional fluid mechanics engineering assistant.

Analyze the user's engineering question using the supplied calculation
results.

USER QUESTION:
{user_query}

CURRENT CALCULATION RESULTS:
{engineering_context}

Instructions:
1. Explain the engineering principle clearly.
2. Use the supplied calculation results where relevant.
3. Show important equations when useful.
4. Include units.
5. Distinguish between calculated results and engineering interpretation.
6. Do not invent missing data.
7. If an assumption is important, state it.
8. Keep the explanation technically accurate and suitable for an
   engineering student or practicing engineer.

Provide a clear, professional engineering response.
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        text = getattr(response, "text", None)

        if not text:
            return {
                "success": False,
                "text": "Gemini returned an empty response."
            }

        return {
            "success": True,
            "text": text,
        }

    except ImportError:
        return {
            "success": False,
            "text": (
                "The Google Gemini package is not installed. "
                "Add the required Google GenAI package to requirements.txt."
            )
        }

    except Exception as exc:
        return {
            "success": False,
            "text": f"Gemini Assistant error: {str(exc)}"
        }
