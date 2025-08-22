# app/services/emotion_service.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "best_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

id2label = {0: "분노", 1: "불안", 2: "슬픔", 3: "평온", 4: "당황", 5: "기쁨"}

def predict_emotion(text: str) -> str:
    """
    사용자 입력 문장을 감정 6종 중 하나로 분류한다.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        pred_id = torch.argmax(outputs.logits, dim=-1).item()
    return id2label[pred_id]
