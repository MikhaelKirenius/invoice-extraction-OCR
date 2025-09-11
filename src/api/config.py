from pydantic import BaseModel
import os
from pathlib import Path

class Settings(BaseModel):
    MODEL_PATH: str = os.getenv("MODEL_PATH", str(Path("models/ner_best_v3").resolve()))
    AGGREGATION_STRATEGY: str = os.getenv("AGGREGATION_STRATEGY", "max")
    CONF_THRESH: float = float(os.getenv("CONF_THRESH", "0.60"))
    ENABLE_OCR: bool = os.getenv("ENABLE_OCR", "true").lower() == "true"
    OCR_LANG: str = os.getenv("OCR_LANG", "en")

settings = Settings()
