from app.services.emotion_service import predict_emotion
from app.services.memory import search_memory
from app.services.summary import summarize_conversation

def classify_emotion(text: str) -> str:
    return predict_emotion(text)

def rag_search(text: str) -> str:
    return search_memory(text)

def summarize(member_id: int) -> str:
    return summarize_conversation(member_id)
