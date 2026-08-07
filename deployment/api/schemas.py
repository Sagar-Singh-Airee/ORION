"""Stable HTTP contract for serving a trained ORION model."""
from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    study_path: str = Field(description="Absolute path to one mounted DICOM study directory")
    report_text: str | None = Field(default=None, max_length=20000)


class PredictResponse(BaseModel):
    study_id: str
    probabilities: dict[str, float]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
