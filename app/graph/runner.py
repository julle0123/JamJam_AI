from app.graph.graph import build_graph
from app.graph.state import ChatState
from app.services.memory import get_user_history
from langchain_core.runnables.history import RunnableWithMessageHistory

async def run_chat(user_input: str, user_id: int, persona: str = "emotional") -> str:
    workflow = build_graph()

    # 히스토리 래핑 (LangChain RunnableWithMessageHistory)
    workflow_with_memory = RunnableWithMessageHistory(
        workflow,
        get_user_history,
        input_messages_key="user_input",
        history_messages_key="history"
    )

    initial_state: ChatState = {
        "user_input": user_input,
        "member_id": user_id,
        "persona": persona
    }

    final_state = workflow_with_memory.invoke(
        initial_state,
        config={"configurable": {"session_id": str(user_id)}}
    )

    return final_state["response"]
