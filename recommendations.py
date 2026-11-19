"""Module 8 — Lifestyle recommendations."""

from preprocessing import encode_yes_no


def recommend(risk_pct: float, user: dict) -> list[str]:
    """Return general educational guidance, independent of model score."""
    tips = [
        "Discuss PCOS concerns or symptoms with a qualified healthcare "
        "professional; this experimental score cannot assess or rule out PCOS."
    ]

    if not encode_yes_no(user.get("exercise", False)):
        tips.append(
            "If appropriate for you, consider regular physical activity "
            "as part of general wellbeing."
        )
    if encode_yes_no(user.get("fast_food", False)):
        tips.append(
            "Consider balanced meals that include vegetables, whole grains, "
            "and minimally processed foods."
        )
    if user.get("stress") in ("high", 2):
        tips.append(
            "For stress, consider relaxation practices such as breathing "
            "exercises or journaling."
        )
    if float(user.get("sleep", 7)) < 7:
        tips.append(
            "Consider discussing ongoing sleep difficulties with a "
            "healthcare professional."
        )
    return tips
