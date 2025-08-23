# app/graph/runner.py
# 그래프 인스턴스 생애주기 관리 + 에이전트 호출 편의 함수.
from typing import Optional
from sqlalchemy.orm import Session
from langchain_core.messages import HumanMessage

from app.graph.graph import build_agent_graph
from app.graph.callbacks import ReactTraceCallback
import logging

# 그래프는 체크포인터 포함으로 모듈 레벨에 1회 생성
_graph = build_agent_graph()

async def run_chat_agent(
    user_input: str,
    user_id: int,
    db: Optional[Session] = None,         # 직렬화 이슈로 state에는 넣지 않음
    session_id: Optional[str] = None,
    force_summary: bool = False,
    disable_preload: bool = False,
    debug_trace: bool = False,            # ← 스트림/툴콜 트레이스 ON
) -> str:
    sid = str(session_id or user_id)

    # debug_trace일 때만 콜백 연결
    callbacks = [ReactTraceCallback(logging.getLogger("react"))] if debug_trace else None

    out = await _graph.ainvoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "member_id": user_id,
            "force_summary": force_summary,
            "disable_preload": disable_preload,
            "debug_trace": debug_trace,
        },
        config={
            "configurable": {"thread_id": sid},
            "tags": ["agent", f"user:{user_id}", f"session:{sid}"],
            "metadata": {
                "member_id": user_id,
                "session_id": sid,
                "force_summary": force_summary,
                "disable_preload": disable_preload,
                "debug_trace": debug_trace,
            },
            "callbacks": callbacks,   # ← 여기서 콜백 주입
        },
    )
    return out.get("response", "")

