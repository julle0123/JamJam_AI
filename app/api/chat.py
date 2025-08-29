# app/api/chat.py
# /chat 엔드포인트. 에이전트 실행 → DB 로그 저장 → 벡터메모리(Qdrant)에도 동시 기록.
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime, timedelta, timezone
import inspect

from app.models.schemas import ChatRequest, ChatResponse
from app.graph.runner import run_chat_agent
from app.models.chat_log import ChatLog
from app.core.db import get_db
from app.services.memory import add_chat_memory
#from app.services.emotion_service import predict_emotion

# KST 고정: 서버/컨테이너 TZ와 무관하게 한국시간 기준 기록용
KST = timezone(timedelta(hours=9))
router = APIRouter()

@router.post("/", response_model=ChatResponse)
async def chat(req: ChatRequest, db: Session = Depends(get_db)):
    # 1) 에이전트 실행: LLM이 도구 사용을 자율 판단. force_summary는 힌트 성격.
    output_text: str = await run_chat_agent(
        user_input=req.input,
        user_id=req.member_id,
        db=db,  # state에는 넣지 않지만, 호출측 인터페이스는 유지
        session_id=req.session_id,
        force_summary=req.force_summary or False,
        disable_preload=req.disable_preload or False,
        debug_trace=req.debug_trace or False,
    )
    # 사용자 발화 감정 분류(실패해도 서비스 흐름 유지)
    user_emotion = None
    try:
        if req.input and req.input.strip():
            user_emotion = predict_emotion(req.input)
    except Exception as e:
        print(f"[emotion] user predict warn: {e}")


    # 2) 관계형 DB에 대화 로그 저장
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
        # DB 오류 시 롤백 후 500 반환
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    # 3) 벡터 메모리에 동시 저장(검색/회상용). 실패해도 서비스 흐름은 유지.
    try:
        if inspect.iscoroutinefunction(add_chat_memory):
            await add_chat_memory(req.member_id, req.input, "user", chat_log.chat_id, created)
            await add_chat_memory(req.member_id, output_text, "bot", chat_log.chat_id, created)
        else:
            add_chat_memory(req.member_id, req.input, "user", chat_log.chat_id, created)
            add_chat_memory(req.member_id, output_text, "bot", chat_log.chat_id, created)
    except Exception as e:
        # 메모리 저장 경고만 출력(치명적 오류 아님)
        print(f"[memory] save warn: {e}")

    # 4) 최종 응답
    return ChatResponse(output=output_text, user_emotion=user_emotion)
