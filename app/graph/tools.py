# app/graph/tools.py
from langchain_core.tools import tool
from sqlalchemy.orm import Session
from app.services.emotion_service import predict_emotion
from app.services.memory import search_memory
from app.services.summary import summarize_conversation
from app.core.db import SessionLocal

@tool
def classify_emotion_tool(text: str) -> str:
    """문장을 6개 감정 중 하나로 분류한다."""
    label = predict_emotion(text)
    return f"emotion={label}"

@tool
def rag_search_tool(query: str, member_id: int, top_k: int = 3) -> str:
    """member_id 필터로 유사 문맥을 검색한다."""
    ctx = search_memory(query, top_k=top_k, member_id=member_id)
    snippet = (ctx[:1500] + " ...") if ctx and len(ctx) > 1500 else (ctx or "")
    return f"ctx_len={len(ctx or '')}\n{snippet}"

@tool
async def summarize_tool(member_id: int, limit: int = 20) -> str:
    """최근 대화(limit) 요약을 생성한다."""
    db: Session = SessionLocal()
    try:
        summary = await summarize_conversation(member_id, db, limit)
        if hasattr(summary, "content"):
            summary = summary.content
        return (summary or "")[:1500]
    except Exception as e:
        return f"summary_error={e}"
    finally:
        db.close()