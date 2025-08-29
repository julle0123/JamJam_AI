# app/services/emotion_service.py
import threading
import torch
from transformers import AutoTokenizer, RobertaForSequenceClassification

MODEL_PATH = "/app/best_model"

_id2label = {0:"분노", 1:"불안", 2:"슬픔", 3:"평온", 4:"당황", 5:"기쁨"}

# lazy-load 대상 (임포트 시점에는 None)
_tokenizer = None
_model = None
_lock = threading.Lock()

def _ensure_loaded():
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return
    with _lock:  # 동시 초기화 방지
        if _tokenizer is None or _model is None:
            tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
            mdl = RobertaForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
            # if torch.cuda.is_available(): mdl.to("cuda")
            mdl.eval()
            _tokenizer, _model = tok, mdl

def predict_emotion(text: str) -> str:
    """요청마다 감정 추론 (모델은 최초 1회만 로드)"""
    _ensure_loaded()
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    # if _model.device.type == "cuda": inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        logits = _model(**inputs).logits
        pred_id = torch.argmax(logits, dim=-1).item()
    return _id2label[pred_id]
