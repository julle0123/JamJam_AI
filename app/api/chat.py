from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from datetime import datetime, timedelta, timezone

from app.models.schemas import ChatRequest, ChatResponse
from app.graph.runner import run_chat
from app.models.chat_log import ChatLog
from app.core.db import get_db
from app.services.memory import add_chat_memory   # 추가

KST = timezone(timedelta(hours=9))
router = APIRouter()

@router.post("/", response_model=ChatResponse)
async def chat(req: ChatRequest, db: Session = Depends(get_db)):
    # GPT/Agent 실행 (응답 생성) — LangGraph 내부에서 db가 필요하면 runner로 전달
    output_text = await run_chat(
        user_input=req.input,
        user_id=req.member_id,
        db=db
    )

    # DB 저장
    created = datetime.now(KST)
    chat_log = ChatLog(
        member_id=req.member_id,
        user_text=req.input,
        bot_text=output_text,
        created_at=created
    )
    db.add(chat_log)
    db.commit()
    db.refresh(chat_log)

    # Qdrant 동시 저장 (user turn / bot turn)
    add_chat_memory(
        member_id=req.member_id,
        text=req.input,
        role="user",
        chat_id=chat_log.chat_id,
        created_at=created,
    )
    add_chat_memory(
        member_id=req.member_id,
        text=output_text,
        role="bot",
        chat_id=chat_log.chat_id,
        created_at=created,
    )

    return ChatResponse(output=output_text)
