"""Generate a synthetic PCOS-like dataset so the project runs end-to-end
without requiring a Kaggle download. Replace dataset/pcos.csv with the
real Kerala-hospitals dataset when you have it (same column names)."""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)


def make_dataset(n: int = 1500) -> pd.DataFrame:
    age = RNG.integers(16, 45, n)
    height = RNG.normal(160, 7, n).clip(140, 185)
    weight = RNG.normal(65, 14, n).clip(40, 120)
    bmi = weight / (height / 100) ** 2

    irregular = RNG.binomial(1, 0.45, n)
    acne = RNG.binomial(1, 0.4, n)
    hair_growth = RNG.binomial(1, 0.35, n)
    weight_gain = (bmi > 25).astype(int) * RNG.binomial(1, 0.7, n)
    skin_dark = RNG.binomial(1, 0.3, n)
    fast_food = RNG.binomial(1, 0.5, n)
    exercise = RNG.binomial(1, 0.4, n)
    stress = RNG.integers(0, 3, n)
    sleep = RNG.normal(6.8, 1.2, n).clip(3, 10)

    score = (
        0.9 * irregular + 0.7 * acne + 0.8 * hair_growth + 0.6 * weight_gain
        + 0.5 * skin_dark + 0.3 * fast_food - 0.5 * exercise
        + 0.3 * stress - 0.2 * (sleep - 7) + 0.05 * (bmi - 24)
    )
    prob = 1 / (1 + np.exp(-(score - 1.6)))
    pcos = (RNG.random(n) < prob).astype(int)

    return pd.DataFrame({
        "age": age, "weight": weight.round(1), "height": height.round(1),
        "bmi": bmi.round(2),
        "irregular_periods": irregular, "acne": acne, "hair_growth": hair_growth,
        "weight_gain": weight_gain, "skin_darkening": skin_dark,
        "fast_food": fast_food, "exercise": exercise,
        "stress": stress, "sleep": sleep.round(1),
        "pcos": pcos,
    })


def ensure_dataset(path: str = "dataset/pcos.csv") -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        make_dataset().to_csv(p, index=False)
    return str(p)


if __name__ == "__main__":
    print("Wrote:", ensure_dataset())
