from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from datetime import datetime, timedelta, timezone

from app.models.schemas import ChatRequest, ChatResponse
from app.graph.runner import run_chat
from app.models.chat_log import ChatLog
from app.core.db import get_db

KST = timezone(timedelta(hours=9))
router = APIRouter()

@router.post("/", response_model=ChatResponse)
async def chat(req: ChatRequest, db: Session = Depends(get_db)):
    # GPT/Agent 실행 (응답 생성)
    output_text = await run_chat(
        user_input=req.input,
        user_id=req.member_id,  # 실제 DB member_id
        db=db
    )

    # 대화 로그 저장
    chat_log = ChatLog(
        member_id=req.member_id,
        user_text=req.input,
        bot_text=output_text,
        created_at=datetime.now(KST)
    )
    db.add(chat_log)
    db.commit()
    db.refresh(chat_log)

    return ChatResponse(output=output_text)
