from transformers import AutoTokenizer, AutoModelForTokenClassification 
from transformers import pipeline
from ..config import settings
from typing import Dict
from src.ocr.preprocessing_text import TextProcessingNER
from src.utils.logger import default_logger as Logger
import os

class ExtractorService:
    def __init__(self):
        Logger.info(f"Loading NER model from: {settings.MODEL_PATH}")
        self.tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_PATH)
        self.model = AutoModelForTokenClassification.from_pretrained(settings.MODEL_PATH)
        self.text_processor = TextProcessingNER(self.model, self.tokenizer)

        self.pipe = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy=settings.AGGREGATION_STRATEGY,
            device_map="auto"
        )
        
    def extract(self, text:str) -> Dict:
        return self.text_processor.extract_entities(text)
    
extractor_service: ExtractorService | None = None

def init_extractor():
    global extractor_service
    Logger.info("Initializing extractor model...") 
    extractor_service = ExtractorService()
    Logger.info("Extractor model initialized.")