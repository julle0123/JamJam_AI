# app/services/summary.py
# 최근 대화 N개를 요약. 읽기 트랜잭션을 COMMIT로 남겨 로그를 깔끔히.
from sqlalchemy.orm import Session
from app.models.chat_log import ChatLog
from app.core.client import llm

async def summarize_conversation(member_id: int, db: Session = None, limit: int = 20) -> str:
    """
    최근 대화를 요약한다.
    - 읽기 트랜잭션을 명시적으로 시작/종료하여 로그가 ROLLBACK 대신 COMMIT로 찍히도록 함.
    """
    if db is None:
        return ""
    
    # 명시적 읽기 트랜잭션(끝날 때 COMMIT) → ROLLBACK 노이즈 제거
    try:
        with db.begin():
            chats = (
                db.query(ChatLog)
                .filter(ChatLog.member_id == member_id)
                .order_by(ChatLog.created_at.desc())
                .limit(limit)
                .all()
            )
    except Exception:
        return ""

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
