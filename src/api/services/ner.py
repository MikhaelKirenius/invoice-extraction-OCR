from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from ..config import settings

class NERService:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_PATH)
        self.model = AutoModelForTokenClassification.from_pretrained(settings.MODEL_PATH)
        self.pipe = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy=settings.AGGREGATION_STRATEGY,
            device_map="auto"
        )

    def predict(self, text: str):
        return self.pipe(text)

ner_service: NERService | None = None

def init_ner():
    global ner_service
    ner_service = NERService()
