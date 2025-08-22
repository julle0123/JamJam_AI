# app/graph/state.py
from typing import TypedDict, List, Optional, Any, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

class AgentState(TypedDict, total=False):
    # LangGraph 권장: 메시지 스트림 하나로 에이전트 루프 구성
    messages: Annotated[List[BaseMessage], add_messages]

    # 세션/식별
    member_id: int
    db: Any

    # 이번 턴 사용자 입력을 확실히 못박기 위한 필드
    current_user_input: str

    # 모델 응답
    response: Optional[str]

    # 프롬프트 구성 요소
    base_system_text: str   # 역할 규칙(고정)
    tool_context: str       # 도구 결과 요약(가변)

