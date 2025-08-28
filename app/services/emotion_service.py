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

SUPPORTED_STYLES = {"Sad", "Embarrassed", "Happy", "Jealous", "Sleepy", "Neutral", "Angry"}

# 3) 우리 한글 라벨 → 스타일(영문) 매핑 (이것만 있으면 됨)
KO2STYLE = {
    "분노": "Angry",
    "불안": "Embarrassed",
    "슬픔": "Sad",
    "평온": "Neutral",
    "당황": "Embarrassed",
    "기쁨": "Happy",
}

def to_style_label_from_ko(label_ko: str) -> str:
    """로컬 모델 한글 라벨을 공급사 스타일(영문)로 변환. 미지정/미지원은 Neutral."""
    style = KO2STYLE.get(label_ko, "Neutral")
    return style if style in SUPPORTED_STYLES else "Neutral"

def decide_style_for_text(text: str) -> str:
    """외부 힌트 없이, 오직 우리 모델 출력만 사용."""
    ko_label = predict_emotion(text)               # "분노/불안/슬픔/평온/당황/기쁨"
    style = to_style_label_from_ko(ko_label)       # → "Angry/Embarrassed/…"