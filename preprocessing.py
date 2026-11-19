
"""Module 2 — Data preprocessing utilities."""

import pandas as pd


YES_NO_COLS = [
    "irregular_periods",
    "acne",
    "hair_growth",
    "weight_gain",
    "skin_darkening",
    "fast_food",
    "exercise",
]

STRESS_MAP = {"low": 0, "medium": 1, "high": 2}

# These features must be in the same order during training and prediction.
# They are available in the real PCOS dataset and the symptom form.
FEATURE_ORDER = [
    "age",
    "weight",
    "height",
    "bmi",
    "irregular_periods",
    "acne",
    "hair_growth",
    "weight_gain",
    "skin_darkening",
    "fast_food",
    "exercise",
]


def calc_bmi(weight_kg: float, height_cm: float) -> float:
    if height_cm <= 0:
        return 0.0

    height_m = height_cm / 100.0
    return round(weight_kg / (height_m * height_m), 2)


def encode_yes_no(value) -> int:
    if isinstance(value, str):
        return int(
            value.strip().lower() in {"yes", "y", "1", "true"}
        )

    return int(bool(value))


def encode_stress(value) -> int:
    """Retained for compatibility with any existing code."""
    if isinstance(value, str):
        return STRESS_MAP.get(value.strip().lower(), 1)

    return int(value)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a dataframe while preserving its existing columns."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.dropna()

    for column in YES_NO_COLS:
        if column in df.columns:
            df[column] = df[column].apply(encode_yes_no)

    if "stress" in df.columns:
        df["stress"] = df["stress"].apply(encode_stress)

    if "bmi" not in df.columns and {
        "weight",
        "height",
    }.issubset(df.columns):
        df["bmi"] = df.apply(
            lambda row: calc_bmi(row["weight"], row["height"]),
            axis=1,
        )

    return df


def build_feature_vector(user_input: dict) -> pd.DataFrame:
    """Convert symptom-form inputs into the model's expected features."""

    weight = float(user_input["weight"])
    height = float(user_input["height"])

    bmi_value = user_input.get("bmi")
    bmi = (
        float(bmi_value)
        if bmi_value is not None
        else calc_bmi(weight, height)
    )

    row = {
        "age": float(user_input["age"]),
        "weight": weight,
        "height": height,
        "bmi": bmi,
        "irregular_periods": encode_yes_no(
            user_input["irregular_periods"]
        ),
        "acne": encode_yes_no(user_input["acne"]),
        "hair_growth": encode_yes_no(user_input["hair_growth"]),
        "weight_gain": encode_yes_no(user_input["weight_gain"]),
        "skin_darkening": encode_yes_no(
            user_input["skin_darkening"]
        ),
        "fast_food": encode_yes_no(user_input["fast_food"]),
        "exercise": encode_yes_no(user_input["exercise"]),
    }

    return pd.DataFrame([row], columns=FEATURE_ORDER, dtype=float)
