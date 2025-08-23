# app/graph/tools.py
import logging
from langchain_core.tools import tool
from sqlalchemy.orm import Session
from app.services.emotion_service import predict_emotion
from app.services.memory import search_memory, recall_or_general_context
from app.services.summary import summarize_conversation
from app.core.db import SessionLocal

logger = logging.getLogger("react")  # 에이전트 트레이스용 로거

@tool
def classify_emotion_tool(text: str) -> str:
    """문장을 6개 감정 중 하나로 분류한다."""
    logger.info("=== REACT / ACTION-INPUT === classify_emotion_tool(text_len=%d)", len(text or ""))  # 본문은 미로그
    label = predict_emotion(text)  # 로컬 모델 추론
    out = f"emotion={label}"       # 에이전트가 파싱하기 쉬운 포맷
    logger.info("=== REACT / OBSERVATION === classify_emotion_tool -> %s", out)  # 관측치 요약
    return out

@tool
def rag_search_tool(query: str, member_id: int, top_k: int = 3) -> str:
    """
    member_id 필터로 유사 문맥 검색.
    '기억/지난번/그때' 등 회상 힌트가 있으면 DB 시간창 확장 회상 모드로 전환.
    - 읽기 트랜잭션을 명시적으로 열어 COMMIT로 종료(로그 깔끔).
    """
    logger.info("=== REACT / ACTION-INPUT === rag_search_tool(member_id=%s, top_k=%s, qlen=%d)",
                member_id, top_k, len(query or ""))  # 본문 미로그
    db: Session = SessionLocal()  # 도구 단위 세션
    try:
        with db.begin():  # 읽기 트랜잭션(ROLLBACK 노이즈 방지)
            ctx = recall_or_general_context(user_input=query, member_id=member_id, db=db, top_k=top_k)
        snippet = (ctx[:1500] + " …") if ctx and len(ctx) > 1500 else (ctx or "")  # 토큰/로그 절약
        logger.info("=== REACT / OBSERVATION === rag_search_tool -> ctx_len=%d", len(ctx or ""))
        return f"ctx_len={len(ctx or '')}\n{snippet}"
    except Exception as e:
        logger.warning("=== REACT / OBSERVATION === rag_search_tool warn: %s", e)
        try:
            ctx = search_memory(query, top_k=top_k, member_id=member_id)  # 폴백: 벡터 검색
            snippet = (ctx[:1500] + " …") if ctx and len(ctx) > 1500 else (ctx or "")
            logger.info("=== REACT / OBSERVATION === rag_search_tool(fallback) -> ctx_len=%d", len(ctx or ""))
            return f"ctx_len={len(ctx or '')}\n{snippet}"
        except Exception as e2:
            logger.error("=== REACT / OBSERVATION === rag_search_tool fallback error: %s", e2)
            return f"ctx_error={e2}"
    finally:
        db.close()  # 세션 정리

@tool
async def summarize_tool(member_id: int, limit: int = 20) -> str:
    """
    최근 대화(limit) 요약을 생성한다.
    - 내부에서 읽기 트랜잭션을 COMMIT로 종료하도록 summary 서비스가 처리.
    """
    logger.info("=== REACT / ACTION-INPUT === summarize_tool(member_id=%s, limit=%s)", member_id, limit)
    db: Session = SessionLocal()
    try:
        summary = await summarize_conversation(member_id, db, limit)  # LLM 요약 호출
        if hasattr(summary, "content"):
            summary = summary.content  # Message 타입 대비
        s = summary or ""
        logger.info("=== REACT / OBSERVATION === summarize_tool -> out_len=%d", len(s))
        return s[:1500]  # 길이 제한(모델/로그 보호)
    except Exception as e:
        logger.error("=== REACT / OBSERVATION === summarize_tool error: %s", e)
        return f"summary_error={e}"
    finally:
        db.close()
