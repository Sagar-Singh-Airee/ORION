"""FastAPI factory. A trained prediction callable is injected at deployment time."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, HTTPException

from .middleware import RequestContextMiddleware
from .schemas import HealthResponse, PredictRequest, PredictResponse

PredictFunction = Callable[[Path, str | None], dict[str, float]]


def create_app(predict: PredictFunction | None = None, model_version: str | None = None) -> FastAPI:
    app = FastAPI(title="ORION Knee MRI API", version=model_version or "unloaded")
    app.add_middleware(RequestContextMiddleware)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok" if predict else "model_not_loaded", model_loaded=predict is not None, model_version=model_version)

    @app.post("/predict", response_model=PredictResponse)
    def predict_study(request: PredictRequest) -> PredictResponse:
        if predict is None:
            raise HTTPException(status_code=503, detail="No trained model is loaded")
        path = Path(request.study_path)
        if not path.is_dir():
            raise HTTPException(status_code=422, detail="study_path must be an existing DICOM directory")
        probabilities = predict(path, request.report_text)
        if not probabilities or any(not 0 <= value <= 1 for value in probabilities.values()):
            raise HTTPException(status_code=500, detail="Predictor returned invalid probabilities")
        return PredictResponse(study_id=path.name, probabilities=probabilities, model_version=model_version or "unknown")

    return app


app = create_app()
