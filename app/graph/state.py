from typing import TypedDict, List, Any

class ChatState(TypedDict):
    user_input: str
    member_id: int
    persona: str
    tool_flags: dict
    emotion: str
    memory: str
    summary: str
    history: List[Any]   # RunnableWithMessageHistory에서 붙는 대화 히스토리
    response: str
