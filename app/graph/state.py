# app/graph/state.py
# LangGraph 상태 타입 정의. messages는 add_messages로 누적 관리.
from typing import TypedDict, List, Optional, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

class AgentState(TypedDict, total=False):
    # 메시지 스트림(히스토리). add_messages 덕분에 자동 누적/머지.
    messages: Annotated[List[BaseMessage], add_messages]

    # 세션 식별
    member_id: int

    # 모델 최종 응답
    response: Optional[str]

    # 프롬프트 구성 요소
    base_system_text: str   # 역할 규칙(고정)
    preload_context: str          # 선주입 컨텍스트(요약/회상/감정)
    tool_context: str       # 도구 결과 요약(가변)

    force_summary: bool
    disable_preload: bool
    debug_trace: bool