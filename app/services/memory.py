# app/services/memory.py
# (a) 인메모리 히스토리 for RunnableWithMessageHistory, (b) Qdrant 벡터 메모리 저장/검색, (c) "회상 모드": 유사 시점 주변 DB 대화창 확장.

from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta, timezone

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.schema import BaseMessage

from sqlalchemy.orm import Session
from app.core.client import vectorstore
from app.models.chat_log import ChatLog

# ★ 추가: Qdrant 필터 모델 사용
from qdrant_client.http import models as qmodels

_store: Dict[str, ChatMessageHistory] = {}

def get_user_history(session_id: str) -> BaseChatMessageHistory:
    key = str(session_id)
    if key not in _store:
        _store[key] = ChatMessageHistory()
    return _store[key]

def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, BaseMessage):
        return value.content or ""
    if hasattr(value, "content"):
        try:
            return getattr(value, "content") or ""
        except Exception:
            pass
    if isinstance(value, str):
        return value
    return str(value)

def add_chat_memory(
    member_id: int,
    text: Any,
    role: str,
    chat_id: Optional[int] = None,
    created_at: Optional[datetime] = None,
) -> None:
    text_str = _to_text(text).strip()
    if not text_str:
        return
    created_utc = _ensure_utc(created_at) if created_at else datetime.now(timezone.utc)

    # member_id를 문자열로 저장 (KEYWORD 인덱스와 호환)
    vectorstore.add_texts(
        texts=[text_str],
        metadatas=[{
            "member_id": str(member_id),
            "role": role,
            "created_at": created_utc.isoformat(),
            "chat_id": chat_id,
        }],
    )

def _member_filter(member_id: Optional[int]) -> Optional[qmodels.Filter]:
    if member_id is None:
        return None
    return qmodels.Filter(
        must=[qmodels.FieldCondition(
            key="member_id",
            match=qmodels.MatchValue(value=str(member_id))
        )]
    )

def search_memory(query: str, top_k: int = 3, member_id: Optional[int] = None) -> str:
    query_str = _to_text(query)
    try:
        filt = _member_filter(member_id)
        results = vectorstore.similarity_search(query_str, k=top_k, filter=filt)
    except Exception as e:
        print(f"[Qdrant] filtered search failed -> fallback. reason: {e}")
        results = vectorstore.similarity_search(query_str, k=top_k)

    if not results:
        return ""
    return "\n".join(doc.page_content for doc in results)

_RECALL_HINTS = [
    "기억나", "기억 해", "그때", "그 일", "그날", "그 순간",
    "지난번", "전에 말했", "예전에 말했", "그 얘기"
]

def _looks_like_recall(text: Any) -> bool:
    t = _to_text(text)
    return any(h in t for h in _RECALL_HINTS)

def _expand_context_window_by_time(
    db: Session,
    member_id: int,
    center_time: datetime,
    minutes: int = 30,
    limit: int = 30,
) -> str:
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

    lines: List[str] = []
    for c in logs:
        when = _ensure_utc(c.created_at).isoformat()
        lines.append(f"[{when}][USER] {c.user_text}")
        lines.append(f"[{when}][BOT ] {c.bot_text}")
    return "\n".join(lines)

def recall_or_general_context(
    user_input: Any,
    member_id: int,
    db: Optional[Session],
    top_k: int = 3,
    recall_window_min: int = 30,
) -> str:
    if not db or not _looks_like_recall(user_input):
        return search_memory(_to_text(user_input), top_k=top_k, member_id=member_id)

    filt = _member_filter(member_id)
    docs_with_scores = vectorstore.similarity_search_with_score(
        _to_text(user_input),
        k=top_k,
        filter=filt,
    )
    if not docs_with_scores:
        return ""

    contexts: List[str] = []
    for doc, _score in docs_with_scores:
        meta = doc.metadata or {}
        created_at_iso = meta.get("created_at")
        if not created_at_iso:
            continue
        try:
            center = datetime.fromisoformat(created_at_iso.replace("Z", "+00:00"))
        except Exception:
            continue

        ctx = _expand_context_window_by_time(
            db=db,
            member_id=member_id,
            center_time=center,
            minutes=recall_window_min,
        )
        if ctx:
            contexts.append(ctx)

    if not contexts:
        return search_memory(_to_text(user_input), top_k=top_k, member_id=member_id)

    return "\n---\n".join(contexts)
