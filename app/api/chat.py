# app/api/chat.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime, timedelta, timezone
import inspect

from app.models.schemas import ChatRequest, ChatResponse
from app.graph.runner import run_chat_agent  # 에이전트 러너
from app.models.chat_log import ChatLog
from app.core.db import get_db
from app.services.memory import add_chat_memory

KST = timezone(timedelta(hours=9))
router = APIRouter()

@router.post("/", response_model=ChatResponse)
async def chat(req: ChatRequest, db: Session = Depends(get_db)):
    output_text: str = await run_chat_agent(
        user_input=req.input,
        user_id=req.member_id,
        db=db,
        session_id=req.session_id,
    )

    created = datetime.now(KST)
    try:
        chat_log = ChatLog(
            member_id=req.member_id,
            user_text=req.input,
            bot_text=output_text,
            created_at=created,
        )
        db.add(chat_log); db.commit(); db.refresh(chat_log)
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    try:
        if inspect.iscoroutinefunction(add_chat_memory):
            await add_chat_memory(req.member_id, req.input, "user", chat_log.chat_id, created)
            await add_chat_memory(req.member_id, output_text, "bot", chat_log.chat_id, created)
        else:
            add_chat_memory(req.member_id, req.input, "user", chat_log.chat_id, created)
            add_chat_memory(req.member_id, output_text, "bot", chat_log.chat_id, created)
    except Exception as e:
        print(f"[memory] save warn: {e}")

    return ChatResponse(output=output_text)
