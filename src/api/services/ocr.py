from ..config import settings
from src.utils.logger import default_logger as Logger

_reader = None

def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        langs = settings.OCR_LANG.split(",")
        _reader = easyocr.Reader(langs)
    return _reader

def ocr_image_to_text(image_bytes: bytes):
    if not settings.ENABLE_OCR:
        return "", {"enabled": False}
    from PIL import Image
    import io, numpy as np

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img)
    reader = _get_reader()
    result = reader.readtext(arr, detail=1, paragraph=False)

    texts, confs = [], []
    for item in result:
        if len(item) == 3:   
            _, text, conf = item
        elif len(item) == 2: 
            text, conf = item
        else:                
            text, conf = str(item), 1.0
        texts.append(text)
        confs.append(conf)
    return " ".join(texts), {
        "n_boxes": len(result),
        "avg_conf": float(sum(confs) / len(confs)) if confs else None,
        "enabled": True
    }
