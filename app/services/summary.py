# app/services/summary.py
from sqlalchemy.orm import Session
from app.models.chat_log import ChatLog
from app.core.client import llm
def summarize_conversation(member_id: int, db: Session = None, limit: int = 20) -> str:
    """
    특정 사용자의 최근 대화를 요약한다.
    """
    if db is None:
        return ""

    chats = (
        db.query(ChatLog)
        .filter(ChatLog.member_id == member_id)
        .order_by(ChatLog.created_at.desc())
        .limit(limit)
        .all()
    )
    if not chats:
        return ""

    # 텍스트 병합
    conversation_text = "\n".join([f"U: {c.user_text}\nB: {c.bot_text}" for c in reversed(chats)])

    # 요약 프롬프트
    prompt = f"""
    아래는 사용자와 챗봇의 대화 기록이다. 
    최근 대화 맥락을 요약해서 제공해라.

    {conversation_text}
    """
    summary = llm.invoke(prompt)
    return summary
