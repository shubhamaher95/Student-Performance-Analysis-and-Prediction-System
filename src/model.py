
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "Student Performance Factors.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "evaluation_results"
MODEL_PATH = PROJECT_DIR / "src" / "student_performance_model.joblib"

FEATURES = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity",
]
TARGET = "Exam_Score"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH).drop_duplicates()
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", Ridge()),
    ])

    search = GridSearchCV(
        pipeline,
        {"regressor__alpha": [0.01, 0.1, 1, 10, 100]},
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    predictions = best_model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2": r2_score(y_test, predictions),
        "Best_Alpha": search.best_params_["regressor__alpha"],
        "CV_MAE": -search.best_score_,
    }

    print("\nSTUDENT PERFORMANCE MODEL RESULTS")
    print("-" * 38)
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")

    pd.DataFrame([metrics]).to_csv(
        OUTPUT_DIR / "standalone_model_metrics.csv", index=False
    )

    pd.DataFrame({
        "Actual_Exam_Score": y_test.to_numpy(),
        "Predicted_Exam_Score": predictions,
    }).to_csv(
        OUTPUT_DIR / "standalone_predictions.csv", index=False
    )

    joblib.dump(best_model, MODEL_PATH)
    print("\nModel and evaluation files saved successfully.")


if __name__ == "__main__":
    main()
