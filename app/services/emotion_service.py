# app/services/emotion_service.py
# 로컬 감정분류 모델 로드/추론. 전역 1회 로드로 성능 안정.

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# MODEL_PATH = "outputs_trainer_final2/best_model"  # 예: 개발 환경
MODEL_PATH = "/app/best_model"                  # 예: 컨테이너/서버 환경

# 전역 1회 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)

id2label = {0: "분노", 1: "불안", 2: "슬픔", 3: "평온", 4: "당황", 5: "기쁨"}

def predict_emotion(text: str) -> str:
    # 단일 문장 감정 분류(배치도 문제없이 동작)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        pred_id = torch.argmax(outputs.logits, dim=-1).item()
    return id2label[pred_id]
