from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    field_id: str
    event_timestamp: str  # ISO-8601, e.g. "2024-06-01T00:00:00"


class PredictResponse(BaseModel):
    field_id: str
    predicted_yield_kg_per_ha: float
    lower_bound: float
    upper_bound: float
    model_version: str


class BatchPredictRequest(BaseModel):
    requests: list[PredictRequest] = Field(..., min_length=1, max_length=500)


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
