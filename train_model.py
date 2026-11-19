
"""Train and evaluate the symptom-based PCOS prototype on real dataset records."""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing import FEATURE_ORDER


DATA_PATH = Path("dataset/PCOS_data_without_infertility.xlsx")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def load_training_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.resolve()}"
        )

    raw = pd.read_excel(DATA_PATH, sheet_name="Full_new")
    raw.columns = raw.columns.str.strip()

    # Map the dataset's column names to our app's feature names.
    column_map = {
        "PCOS (Y/N)": "pcos",
        "Age (yrs)": "age",
        "Weight (Kg)": "weight",
        "Height(Cm)": "height",
        "BMI": "bmi",
        "Cycle(R/I)": "cycle_code",
        "Weight gain(Y/N)": "weight_gain",
        "hair growth(Y/N)": "hair_growth",
        "Skin darkening (Y/N)": "skin_darkening",
        "Pimples(Y/N)": "acne",
        "Fast food (Y/N)": "fast_food",
        "Reg.Exercise(Y/N)": "exercise",
    }

    missing_columns = [
        name for name in column_map if name not in raw.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing expected dataset columns: {missing_columns}"
        )

    df = raw[list(column_map)].rename(columns=column_map).copy()

    # Convert selected columns to numeric values.
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # In this dataset, 2 means regular and 4 means irregular.
    # Other cycle codes are treated as invalid, not guessed.
    df["irregular_periods"] = df["cycle_code"].map({
        2: 0,
        4: 1,
    })

    rows_before = len(df)

    # Remove records missing required values or containing invalid labels.
    df = df.dropna(subset=["pcos"] + FEATURE_ORDER)
    df = df[df["pcos"].isin([0, 1])].copy()

    df["pcos"] = df["pcos"].astype(int)

    # Never use the cycle code itself as an additional model feature.
    df = df.drop(columns=["cycle_code"])

    print("Original records:", rows_before)
    print("Records used:", len(df))
    print("Records excluded:", rows_before - len(df))
    print("\nClass counts:")
    print(
        df["pcos"]
        .value_counts()
        .sort_index()
        .rename(index={0: "No PCOS (0)", 1: "PCOS (1)"})
        .to_string()
    )

    if df.empty or df["pcos"].nunique() != 2:
        raise ValueError(
            "Training data must contain records from both target classes."
        )

    return df[FEATURE_ORDER], df["pcos"]


def train():
    X, y = load_training_data()

    # Keep test records separate from all model selection.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=42)),
    ])

    parameter_grids = [
        {
            "clf": [
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                )
            ],
            "clf__C": [0.01, 0.1, 1, 10],
        },
        {
            "clf": [
                RandomForestClassifier(
                    class_weight="balanced",
                    random_state=42,
                )
            ],
            "clf__n_estimators": [150, 300],
            "clf__max_depth": [None, 5, 10],
            "clf__min_samples_leaf": [1, 3],
        },
        {
            "clf": [
                GradientBoostingClassifier(random_state=42)
            ],
            "clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth": [1, 2],
        },
    ]

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grids,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    print("\nComparing models using 5-fold cross-validation...")
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    predictions = best_model.predict(X_test)
    probabilities = best_model.predict_proba(X_test)[:, 1]

    # Compare against a simple majority-class baseline.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)

    print("\nBest model:", type(best_model.named_steps["clf"]).__name__)
    print("Best settings:", search.best_params_)
    print(
        "Best cross-validation ROC-AUC:",
        round(search.best_score_, 3),
    )

    print("\n--- Held-out test results ---")
    print("Majority baseline accuracy:",
          round(accuracy_score(y_test, baseline_predictions), 3))
    print("Model test accuracy:",
          round(accuracy_score(y_test, predictions), 3))
    print("Balanced accuracy:",
          round(balanced_accuracy_score(y_test, predictions), 3))
    print("Test ROC-AUC:",
          round(roc_auc_score(y_test, probabilities), 3))

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["No PCOS (0)", "PCOS (1)"],
        zero_division=0,
    ))

    print("Confusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    output_path = MODELS_DIR / "symptom_model.joblib"
    joblib.dump(best_model, output_path)
    print(f"\nSaved selected model to: {output_path}")


if __name__ == "__main__":
    train()
