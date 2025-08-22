# app/services/summary.py
from sqlalchemy.orm import Session
from app.models.chat_log import ChatLog
from app.core.client import llm

async def summarize_conversation(member_id: int, db: Session = None, limit: int = 20) -> str:
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

    conversation_text = "\n".join([f"U: {c.user_text}\nB: {c.bot_text}" for c in reversed(chats)])

    prompt = f"""
아래는 사용자와 챗봇의 대화 기록이다.
최근 대화 맥락을 5줄 이내로 핵심만 요약하라.

{conversation_text}
"""
    msg = await llm.ainvoke(prompt, config={"run_name": "Summarize"})
    return getattr(msg, "content", str(msg))
