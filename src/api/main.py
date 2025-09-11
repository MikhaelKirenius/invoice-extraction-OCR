from fastapi import FastAPI, UploadFile, File, HTTPException, Request 
from fastapi.middleware.cors import CORSMiddleware
from .schemas import HealthResponse, PredictTextRequest, BulkResponse, BulkResult
from .services.ocr import ocr_image_to_text
from .config import settings
from src.utils.logger import default_logger as Logger
from typing import List

from contextlib import asynccontextmanager
from .services.extractor import init_extractor, ExtractorService, extractor_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.extractor_service = ExtractorService()
    yield

app = FastAPI(title="Invoice NER API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
def health():
    
    return HealthResponse(
        model_loaded= app.state.extractor_service is not None,
        ocr_enabled=settings.ENABLE_OCR
    )

@app.post("/predict-text")
def predict_text(payload: PredictTextRequest):
    if extractor_service is None:
        raise HTTPException(503, "Model not loaded")
    structured = extractor_service.extract(payload.text)
  
    return {
        "structured": structured,
        "meta": {"source": "text"}
    }

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...), request: Request = None):
    extractor_service = getattr(request.app.state, "extractor_service", None)
    if extractor_service is None:
        raise HTTPException(503, "Model not loaded")
    if not settings.ENABLE_OCR:
        raise HTTPException(400, "OCR disabled")

    text, ocr_meta = ocr_image_to_text(await file.read())
    if not text.strip():
        raise HTTPException(422, "OCR produced empty text")

    structured = extractor_service.extract(text)
    Logger.info(f"Hasil ekstraksi: {structured}")
    return {
        "structured": structured,
        "meta": {"source": "image", "ocr": ocr_meta}
    }

@app.post("/predict-images", response_model=BulkResponse)
async def predict_images(files: List[UploadFile] = File(...), request: Request = None):
    extractor_service: ExtractorService = getattr(request.app.state, "extractor_service", None)
    if extractor_service is None:
        raise HTTPException(503, "Model not loaded")
    if not settings.ENABLE_OCR:
        raise HTTPException(400, "OCR disabled")

    results: List[BulkResult] = []
    for f in files:
        try:
            content = await f.read()
            text, ocr_meta = ocr_image_to_text(content)
            if not text.strip():
                raise ValueError("OCR produced empty text")
            structured = extractor_service.extract(text)
            results.append(BulkResult(filename=f.filename, structured=structured, ocr_meta=ocr_meta))
        except Exception as e:
            results.append(BulkResult(filename=f.filename, error=str(e)))

    return BulkResponse(results=results)