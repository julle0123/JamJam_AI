# app/api/tts.py
from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime, timedelta, timezone
import inspect

from app.models.schemas import ChatRequest
from app.graph.runner import run_chat_agent
from app.models.chat_log import ChatLog
from app.core.db import get_db
from app.services.memory import add_chat_memory
from app.services.tts_service import supertone_tts, TTSServiceError
from app.core.config import settings  
from app.services.emotion_service import decide_style_for_text

KST = timezone(timedelta(hours=9))
router = APIRouter(tags=["chat-tts"])


def decide_style_for_text(text: str) -> str:
    ko_label = predict_emotion(text)
    style = to_style_label_from_ko(ko_label)
    return style

@router.post("/chat/tts")
async def chat_tts(req: ChatRequest, db: Session = Depends(get_db)):
    # 1) LLM 실행
    output_text: str = await run_chat_agent(
        user_input=req.input,
        user_id=req.member_id,
        db=db,
        session_id=req.session_id,
        force_summary=req.force_summary or False,
        disable_preload=req.disable_preload or False,
        debug_trace=req.debug_trace or False,
    )

    # 2) DB 저장
    created = datetime.now(KST)
    try:
        chat_log = ChatLog(
            member_id=req.member_id,
            user_text=req.input,
            bot_text=output_text,
            created_at=created,
        )
        db.add(chat_log)
        db.commit()
        db.refresh(chat_log)
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    # 3) 벡터 메모리
    try:
        if inspect.iscoroutinefunction(add_chat_memory):
            await add_chat_memory(req.member_id, req.input, "user", chat_log.chat_id, created)
            await add_chat_memory(req.member_id, output_text, "bot", chat_log.chat_id, created)
        else:
            add_chat_memory(req.member_id, req.input, "user", chat_log.chat_id, created)
            add_chat_memory(req.member_id, output_text, "bot", chat_log.chat_id, created)
    except Exception as e:
        print(f"[memory] save warn: {e}")

    # 4) 내부 기본값으로 TTS 호출
    try:
        emotion = detect_emotion_from_text(output_text)  # "neutral"
        audio_bytes, mime = await supertone_tts(
            text=output_text,
            voice=settings.TTS_DEFAULT_VOICE,
            emotion=emotion,
            fmt=settings.TTS_DEFAULT_FORMAT,   # "mp3"
            sample_rate=None,                  # 필요시 settings에 기본값 추가 가능
            speed=1.0,                         # 필요시 settings에 기본값 추가 가능
        )
    except TTSServiceError as e:
        raise HTTPException(status_code=502, detail=str(e))

    filename_ext = settings.TTS_DEFAULT_FORMAT.lower() or "mp3"
    headers = {"Content-Disposition": f'inline; filename="reply.{filename_ext}"'}
    return Response(content=audio_bytes, media_type=mime, headers=headers)
