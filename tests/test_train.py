import json

import numpy as np
import pandas as pd
from src.train import check_label_drift, find_best_threshold, train


FEATURE_NAMES = [
    "age",
    "workclass",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]


def _make_temp_data(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    features = rng.random((n, len(FEATURE_NAMES)))
    target = rng.integers(0, 2, size=n)
    data = pd.DataFrame(features, columns=FEATURE_NAMES)
    data["target"] = target

    train_path = tmp_path / "train.csv"
    eval_path = tmp_path / "holdout.csv"
    data.iloc[:160].to_csv(train_path, index=False)
    data.iloc[160:].to_csv(eval_path, index=False)
    return train_path, eval_path


def _train_in_tmp_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train_path, eval_path = _make_temp_data(tmp_path)
    return train(
        {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2},
        data_path=str(train_path),
        eval_path=str(eval_path),
    )


def test_train_returns_float(tmp_path, monkeypatch):
    f1 = _train_in_tmp_path(tmp_path, monkeypatch)
    assert isinstance(f1, float)
    assert 0.0 <= f1 <= 1.0


def test_report_file_created(tmp_path, monkeypatch):
    _train_in_tmp_path(tmp_path, monkeypatch)

    report_path = tmp_path / "outputs" / "report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert set(report) == {
        "f1_score",
        "accuracy",
        "best_threshold",
        "best_threshold_f1",
        "train_positive_rate",
    }
    assert all(0.0 <= value <= 1.0 for value in report.values())
    # Quet nguong co ca 0.5 nen F1 tot nhat khong the thap hon F1 mac dinh
    assert report["best_threshold_f1"] >= report["f1_score"]


def test_detail_report_created(tmp_path, monkeypatch):
    _train_in_tmp_path(tmp_path, monkeypatch)

    detail = (tmp_path / "outputs" / "detail.txt").read_text(encoding="utf-8")
    assert "Confusion matrix" in detail
    assert "precision" in detail and "recall" in detail


def test_best_threshold_beats_or_matches_default():
    y_true = pd.Series([0, 0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.35, 0.4, 0.9])
    threshold, best_f1 = find_best_threshold(y_true, probabilities)
    assert threshold == 0.4
    assert best_f1 == 1.0


def test_check_label_drift_warns_when_rate_is_far_from_reference(capsys):
    assert check_label_drift(pd.Series([1, 1, 0, 0])) == 0.5
    assert "DATA DRIFT" in capsys.readouterr().out


def test_model_file_created(tmp_path, monkeypatch):
    _train_in_tmp_path(tmp_path, monkeypatch)
    assert (tmp_path / "models" / "model.joblib").is_file()
