from __future__ import annotations
from typing import Dict, List, Optional
from datetime import datetime, timedelta, timezone
import re

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from sqlalchemy.orm import Session
from app.core.client import vectorstore
from app.models.chat_log import ChatLog

# -------- In-memory history for RunnableWithMessageHistory --------
_store: Dict[str, ChatMessageHistory] = {}

def get_user_history(user_id: str) -> BaseChatMessageHistory:
    if user_id not in _store:
        _store[user_id] = ChatMessageHistory()
    return _store[user_id]

# -------- Qdrant 저장 --------
def _ensure_utc(dt: datetime) -> datetime:
    # DB에서 tz-naive로 올 수 있으니 UTC로 표준화
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def add_chat_memory(
    member_id: int,
    text: str,
    role: str,                      # "user" | "bot"
    chat_id: Optional[int] = None,
    created_at: Optional[datetime] = None,
) -> None:
    """한 턴의 대화를 Qdrant(VectorStore)에 저장."""
    if not text:
        return
    created_utc = _ensure_utc(created_at) if created_at else datetime.now(timezone.utc)

    vectorstore.add_texts(
        texts=[text],
        metadatas=[{
            "member_id": member_id,
            "role": role,
            "created_at": created_utc.isoformat(),
            "chat_id": chat_id,
        }],
    )

# -------- 일반 유사도 검색 --------
def search_memory(query: str, top_k: int = 3, member_id: Optional[int] = None) -> str:
    if member_id is None:
        results = vectorstore.similarity_search(query, k=top_k)
    else:
        results = vectorstore.similarity_search(
            query, k=top_k,
            filter={"must": [{"key": "member_id", "match": {"value": member_id}}]}
        )
    if not results:
        return ""
    return "\n".join([doc.page_content for doc in results])

# -------- “그때 그 일 기억나?” → 사건 앵커 기반 회상 --------
_RECALL_HINTS = [
    "기억나", "기억 해", "그때", "그 일", "그날", "그 순간",
    "지난번", "전에 말했", "예전에 말했", "그 얘기"
]

def _looks_like_recall(text: str) -> bool:
    t = text or ""
    return any(h in t for h in _RECALL_HINTS)

def _expand_context_window_by_time(
    db: Session,
    member_id: int,
    center_time: datetime,
    minutes: int = 30,
    limit: int = 30,
) -> str:
    """중심 시각 주변의 대화(turn window)로 맥락을 확장."""
    start = center_time - timedelta(minutes=minutes)
    end = center_time + timedelta(minutes=minutes)

    logs: List[ChatLog] = (
        db.query(ChatLog)
        .filter(ChatLog.member_id == member_id)
        .filter(ChatLog.created_at >= start)
        .filter(ChatLog.created_at <= end)
        .order_by(ChatLog.created_at.asc())
        .limit(limit)
        .all()
    )
    if not logs:
        return ""

    lines = []
    for c in logs:
        when = _ensure_utc(c.created_at).isoformat()
        lines.append(f"[{when}][USER] {c.user_text}")
        lines.append(f"[{when}][BOT ] {c.bot_text}")
    return "\n".join(lines)

def recall_or_general_context(
    user_input: str,
    member_id: int,
    db: Optional[Session],
    top_k: int = 3,
    recall_window_min: int = 30,
) -> str:
    """
    - 사용자가 회상 의도가 보이면: Qdrant에서 member_id 필터로 상위 1~3개 사건 앵커 검색
      → 각 결과의 created_at을 DB 기준으로 주변 window로 확장해서 맥락 구성
    - 아니면 일반 RAG
    """
    if not db or not _looks_like_recall(user_input):
        # 일반 RAG (member_id로 스코프 제한)
        return search_memory(user_input, top_k=top_k, member_id=member_id)

    # 회상: 앵커 후보 검색 (유사도 상위)
    docs_with_scores = vectorstore.similarity_search_with_score(
        user_input, k=top_k,
        filter={"must": [{"key": "member_id", "match": {"value": member_id}}]}
    )
    if not docs_with_scores:
        return ""

    # 각 앵커에 대해 주변 대화 윈도우를 모아서 하나의 컨텍스트로
    contexts: List[str] = []
    for doc, _score in docs_with_scores:
        meta = doc.metadata or {}
        created_at_iso = meta.get("created_at")
        try:
            center = datetime.fromisoformat(created_at_iso.replace("Z", "+00:00"))
        except Exception:
            # 메타데이터가 없거나 파싱 실패하면 해당 앵커 스킵
            continue
        ctx = _expand_context_window_by_time(
            db=db,
            member_id=member_id,
            center_time=center,
            minutes=recall_window_min,
        )
        if ctx:
            contexts.append(ctx)

    # 앵커 기반 확장 결과가 없으면 일반 RAG로 대체
    if not contexts:
        return search_memory(user_input, top_k=top_k, member_id=member_id)

    # 여러 앵커가 있으면 이어붙여 제공
    return "\n---\n".join(contexts)
