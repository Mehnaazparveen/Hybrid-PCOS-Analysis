"""Module 8 — Lifestyle recommendations."""

def recommend(risk_pct: float, user: dict) -> list[str]:
    tips: list[str] = []
    if risk_pct >= 60:
        tips.append("Consult a gynecologist for clinical evaluation.")
    elif risk_pct >= 40:
        tips.append("Monitor symptoms and re-check in 4-6 weeks.")
    else:
        tips.append("Maintain current healthy routine.")

    bmi = user.get("bmi", 0)
    if bmi >= 25:
        tips.append("Aim for a 5-10% weight reduction via balanced diet.")
    if not user.get("exercise"):
        tips.append("Add 30 min moderate exercise (walking, yoga) 5 days/week.")
    if user.get("fast_food"):
        tips.append("Reduce processed/fast food; prefer whole grains & greens.")
    if user.get("stress") in ("high", 2):
        tips.append("Practice daily stress relief: meditation, journaling, breathing.")
    if float(user.get("sleep", 7)) < 7:
        tips.append("Improve sleep hygiene; target 7-8 hours nightly.")
    tips.append("Hydrate well and limit refined sugar to support insulin balance.")
    return tips
