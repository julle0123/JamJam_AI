# app/graph/nodes.py
from typing import List, Optional
from langchain_core.messages import (
    BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
)
from app.graph.prompts import load_prompt_template
from app.core.client import agent_llm
from app.graph.tools import classify_emotion_tool, rag_search_tool, summarize_tool

# ---- LLM + Tools ----
LLM_WITH_TOOLS = agent_llm().bind_tools(
    [classify_emotion_tool, rag_search_tool, summarize_tool]
).with_config({"run_name": "AgentWithTools"})

# ---- 역할 프롬프트 ----
def _ensure_role_text(member_id: Optional[int]) -> str:
    role = load_prompt_template("role")
    mi = "" if member_id is None else str(member_id)
    return f"""{role}

[고정 파라미터]
- member_id = {mi}

[도구 호출 가이드]
- rag_search_tool(query="{{사용자요청}}", member_id={mi}, top_k=3)
- summarize_tool(member_id={mi}, limit=20)
- classify_emotion_tool(text="{{사용자요청}}")
"""

def _collect_recent_tool_msgs(messages: List[BaseMessage]) -> List[ToolMessage]:
    """가장 최근 AI 이후 ToolMessage들만 모음."""
    idx = None
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            idx = i
            break
    if idx is None:
        return []
    out: List[ToolMessage] = []
    for m in messages[idx + 1:]:
        if isinstance(m, ToolMessage):
            out.append(m)
        elif isinstance(m, AIMessage):
            break
    return out

def _make_tool_section(tool_msgs: List[ToolMessage]) -> str:
    if not tool_msgs:
        return ""
    lines: List[str] = []
    for tm in tool_msgs:
        name = getattr(tm, "name", "tool")
        out = (tm.content or "").strip()
        if len(out) > 800:
            out = out[:800] + " …"
        lines.append(f"- {name}: {out}")
    return "\n".join(lines)

FOCUS_GUIDE = """[응답 지침]
- 반드시 '사용자 입력'에 직접 대답한다.
- 필요할 때만 '도구 결과'를 답변에 포함한다.
- 주제에서 벗어난 말은 하지 않는다.
- 사용자의 문장을 그대로 복창하지 말고 의미만 간단히 반영한다.
- 입력에 ‘놀자/놀이/게임’이 없으면 먼저 놀이 제안을 하지 않는다.
"""

# ---- LangGraph 노드 ----
async def call_agent(state):
    """
    프롬프트 조립 순서(중요):
    1) [사용자 입력]
    2) [도구 결과]  (있으면)
    3) [역할 규칙]  (base_system_text)
    4) [응답 지침]  (FOCUS_GUIDE)
    -> 위 내용을 단일 SystemMessage로 만든 뒤, 마지막에 HumanMessage를 붙여 LLM 호출
    """
    messages: List[BaseMessage] = state.get("messages", [])
    member_id = state.get("member_id")

    # 역할 텍스트 준비(한 번만)
    if not state.get("base_system_text"):
        state["base_system_text"] = _ensure_role_text(member_id)

    # 이번 턴 사용자 입력
    current_user_input = state.get("current_user_input", "")
    # 방금 실행된 도구 결과(이전 단계에서 요약 저장)
    tool_context = (state.get("tool_context") or "").strip()

    # ---- 단일 SystemMessage로 조립 (순서 강제) ----
    sys_chunks = [f"[사용자 입력]\n{current_user_input or '(없음)'}"]
    if tool_context:
        sys_chunks.append("[도구 결과]\n(아래 정보는 반드시 참고)\n" + tool_context)
    sys_chunks.append("[역할 규칙]\n" + state["base_system_text"])
    sys_chunks.append(FOCUS_GUIDE)

    system_block = "\n\n".join(sys_chunks)

    # 이번 턴 호출 메시지
    final_messages: List[BaseMessage] = [
        SystemMessage(content=system_block),
        HumanMessage(content=current_user_input or ""),
    ]

    ai = await LLM_WITH_TOOLS.ainvoke(final_messages)

    # 기록 저장(메모리용)
    state["messages"] = messages + [*final_messages, ai]
    return state

def should_call_tools(state) -> str:
    msgs: List[BaseMessage] = state.get("messages", [])
    last = msgs[-1] if msgs else None
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "finalize"

async def tools_to_prompt(state):
    """ToolNode 실행 직후 ToolMessage들을 요약 → state['tool_context']에 저장."""
    tmsgs = _collect_recent_tool_msgs(state.get("messages", []))
    section = _make_tool_section(tmsgs)
    state["tool_context"] = section  # call_agent에서 SystemMessage로 합쳐 넣음
    return state

async def finalize(state):
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage) and m.content:
            state["response"] = m.content.strip()
            break
    if not state.get("response"):
        state["response"] = "음… 뭐라고 말해줄까 생각 중이야."
    return state
