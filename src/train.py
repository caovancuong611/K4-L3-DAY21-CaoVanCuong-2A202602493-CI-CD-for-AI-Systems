import json
import os

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

F1_THRESHOLD = 0.65

# Bonus 5: ty le lop duong tham chieu cua bo Adult va muc lech cho phep
REFERENCE_POSITIVE_RATE = 0.248
DRIFT_TOLERANCE = 0.05


def check_label_drift(y_train: pd.Series) -> float:
    """Return the positive-class rate and warn when it drifts from the reference."""
    positive_rate = float(y_train.mean())
    if abs(positive_rate - REFERENCE_POSITIVE_RATE) > DRIFT_TOLERANCE:
        print(
            f"WARNING: DATA DRIFT - positive rate {positive_rate:.1%} differs from "
            f"reference {REFERENCE_POSITIVE_RATE:.1%} by more than "
            f"{DRIFT_TOLERANCE:.0%} points"
        )
    else:
        print(f"Positive rate {positive_rate:.1%} (reference {REFERENCE_POSITIVE_RATE:.1%})")
    return positive_rate


def find_best_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, float]:
    """Bonus 2: sweep decision thresholds 0.10 -> 0.90 and return the one with the best F1."""
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.round(np.arange(0.10, 0.901, 0.05), 2):
        score = f1_score(y_true, (probabilities >= threshold).astype(int))
        if score > best_f1:
            best_threshold, best_f1 = float(threshold), float(score)
    return best_threshold, best_f1


def write_detail_report(y_true: pd.Series, predictions: np.ndarray, path: str) -> None:
    """Bonus 3: confusion matrix and per-class precision/recall as plain text."""
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    report = classification_report(
        y_true,
        predictions,
        labels=[0, 1],
        target_names=["thu_nhap_thap (0)", "thu_nhap_cao (1)"],
        digits=4,
        zero_division=0,
    )
    text = (
        "Confusion matrix (rows = actual, columns = predicted)\n"
        "               pred_0   pred_1\n"
        f"actual_0     {tn:>7}  {fp:>7}\n"
        f"actual_1     {fn:>7}  {tp:>7}\n\n"
        f"{report}"
    )
    with open(path, "w", encoding="utf-8") as detail_file:
        detail_file.write(text)
    print(text)


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """Train a model, record its metrics in MLflow, and save its artifacts."""
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    positive_rate = check_label_drift(y_train)

    with mlflow.start_run():
        mlflow.log_params(params)
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_eval)
        f1 = float(f1_score(y_eval, predictions))
        accuracy = float(accuracy_score(y_eval, predictions))

        best_threshold, best_threshold_f1 = find_best_threshold(
            y_eval, model.predict_proba(X_eval)[:, 1]
        )

        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("best_threshold_f1", best_threshold_f1)
        mlflow.log_metric("train_positive_rate", positive_rate)
        mlflow.sklearn.log_model(model, "model")

        print(f"F1: {f1:.4f} | Accuracy: {accuracy:.4f}")
        print(
            f"Best threshold: {best_threshold:.2f} -> F1 {best_threshold_f1:.4f} "
            f"(F1 at 0.50: {f1:.4f})"
        )

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/report.json", "w", encoding="utf-8") as report_file:
            json.dump(
                {
                    "f1_score": f1,
                    "accuracy": accuracy,
                    "best_threshold": best_threshold,
                    "best_threshold_f1": best_threshold_f1,
                    "train_positive_rate": positive_rate,
                },
                report_file,
            )
        write_detail_report(y_eval, predictions, "outputs/detail.txt")
        mlflow.log_artifact("outputs/detail.txt")

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    return f1


if __name__ == "__main__":
    # Mac dinh ghi vao sqlite:///mlflow.db de khop voi lenh `mlflow ui` o Buoc 1
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    with open("params.yaml", encoding="utf-8") as params_file:
        train(yaml.safe_load(params_file))
