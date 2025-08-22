# app/graph/runner.py
from typing import Optional
from sqlalchemy.orm import Session
from langchain_core.runnables.history import RunnableWithMessageHistory
from app.graph.graph import build_agent_graph
from app.services.memory import get_user_history

async def run_chat_agent(
    user_input: str,
    user_id: int,
    db: Optional[Session] = None,
    session_id: Optional[str] = None,
) -> str:
    graph = build_agent_graph()

    agent_with_history = RunnableWithMessageHistory(
        graph,
        get_user_history,
        input_messages_key="messages",
        history_messages_key="messages",
    )

    sid = str(session_id or user_id)

    out = await agent_with_history.ainvoke(
        # 이번 턴 입력은 반드시 current_user_input에 넣고,
        # messages는 히스토리 저장용으로 그래프가 반환하는 값을 사용하게 둔다.
        {"current_user_input": user_input, "member_id": user_id},
        config={
            "configurable": {"session_id": sid},
            "tags": ["agent", f"user:{user_id}", f"session:{sid}"],
            "metadata": {"member_id": user_id, "session_id": sid},
        },
    )
    return out.get("response", "")
