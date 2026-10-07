import pytest
from fastapi import HTTPException

import src.serve as serve


class StubModel:
    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, features):
        assert len(features) == 1
        assert len(features[0]) == 10
        return [self.prediction]


def test_healthz():
    assert serve.healthz() == {"status": "ok"}


def test_download_model_fetches_configured_s3_object(tmp_path, monkeypatch):
    requested = {}

    class FakeS3Client:
        def download_file(self, bucket, key, path):
            requested.update(bucket=bucket, key=key, path=path)

    monkeypatch.setenv("ARTIFACT_BUCKET", "lab-bucket")
    monkeypatch.setattr(serve, "MODEL_PATH", str(tmp_path / "models" / "model.joblib"))
    monkeypatch.setattr(serve.boto3, "client", lambda service: FakeS3Client())

    serve.download_model()

    assert requested == {
        "bucket": "lab-bucket",
        "key": "artifacts/current/model.joblib",
        "path": str(tmp_path / "models" / "model.joblib"),
    }
    assert (tmp_path / "models").is_dir()


@pytest.mark.parametrize(
    ("prediction", "expected_label"),
    [(0, "thu_nhap_thap"), (1, "thu_nhap_cao")],
)
def test_score_returns_prediction_and_label(monkeypatch, prediction, expected_label):
    monkeypatch.setattr(serve, "model", StubModel(prediction))
    result = serve.score(serve.ScoreRequest(features=[0.0] * 10))
    assert result == {"prediction": prediction, "label": expected_label}


def test_score_rejects_wrong_number_of_features(monkeypatch):
    monkeypatch.setattr(serve, "model", StubModel(0))
    with pytest.raises(HTTPException) as error:
        serve.score(serve.ScoreRequest(features=[0.0] * 9))
    assert error.value.status_code == 400


def test_score_fails_explicitly_when_model_is_not_loaded(monkeypatch):
    monkeypatch.setattr(serve, "model", None)
    with pytest.raises(HTTPException) as error:
        serve.score(serve.ScoreRequest(features=[0.0] * 10))
    assert error.value.status_code == 503
