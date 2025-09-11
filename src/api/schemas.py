from pydantic import BaseModel, Field
from typing import List, Dict, Any,Optional

class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool = True
    ocr_enabled: bool = True

class PredictTextRequest(BaseModel):
    text: str = Field(..., description="teks hasil OCR / input manual")

class EntitySpan(BaseModel):
    entity_group: str
    word: str
    start: int
    end: int
    score: float

class PredictResponse(BaseModel):
    raw_entities: List[EntitySpan]
    structured: Dict[str, Any]
    meta: Dict[str, Any]

class OCRResponse(BaseModel):
    text: str
    meta: Dict[str, Any]

class BulkResult(BaseModel):
    filename: str
    structured: Dict[str, Any] = {}
    ocr_meta: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class BulkResponse(BaseModel):
    results: List[BulkResult]