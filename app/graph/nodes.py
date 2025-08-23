# app/graph/nodes.py
import json
import logging
import asyncio
from typing import List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.graph.prompts import load_prompt_template
from app.core.client import agent_llm
from app.graph.tools import classify_emotion_tool, rag_search_tool, summarize_tool

from app.core.db import SessionLocal
from app.services.summary import summarize_conversation
from app.services.memory import recall_or_general_context
from app.services.emotion_service import predict_emotion

logger = logging.getLogger("react")

LLM_WITH_TOOLS = agent_llm().bind_tools(
    [classify_emotion_tool, rag_search_tool, summarize_tool],
    tool_choice="auto",
).with_config({"run_name": "AgentWithTools"})

AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "{role_text}"),
    ("system",
     "[ReAct 지침]\n"
     "- 필요 시 도구를 선택해 순차 호출(Action)하고, 결과(Observation)를 반영해 다음 결정을 내린다.\n"
     "- 아래 선주입 요약/회상/감정은 문자열 컨텍스트이며, 실제 Tool 실행 결과가 아니다.\n"
     "- 동일 목적의 도구 호출은 금지. 이 턴에서 도구 호출은 최대 1회만 허용.\n"
     "- 최종 답변은 'Final:'로 시작하며 1~2문장 + 되묻기 1문장으로 작성한다."),
    ("system", "{control_hint}"),
    MessagesPlaceholder(variable_name="history"),
    ("system", "선주입 컨텍스트(문자열):\n{preload_context}"),
    ("system", "도구 결과 요약(이전 턴):\n{tool_context}"),
])

def _ensure_role_text(member_id: Optional[int]) -> str:
    try:
        role = load_prompt_template("role")
    except Exception:
        role = ("당신은 3~7세 남자아이 역할의 챗봇이다. 단순 어휘. 정체성/장황/이모지 금지.")
    mi = "" if member_id is None else str(member_id)
    return (
        f"{role}\n\n"
        "[도구]\n"
        f"- classify_emotion_tool(text)\n"
        f"- summarize_tool(member_id={mi}, limit=20)\n"
        f"- rag_search_tool(query, member_id={mi}, top_k=3)\n"
    )

def _history_all(messages: List[BaseMessage]) -> List[BaseMessage]:
    return [m for m in messages if isinstance(m, (HumanMessage, AIMessage, ToolMessage))]

def _collect_recent_tool_msgs(messages: List[BaseMessage]) -> List[ToolMessage]:
    idx = None
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            idx = i; break
    if idx is None:
        return []
    out: List[ToolMessage] = []
    for m in messages[idx + 1:]:
        if isinstance(m, ToolMessage): out.append(m)
        elif isinstance(m, AIMessage): break
    return out

def _summarize_tools(tool_msgs: List[ToolMessage]) -> str:
    if not tool_msgs:
        return ""
    lines = []
    for tm in tool_msgs:
        name = getattr(tm, "name", "tool")
        out = (tm.content or "").strip()
        if len(out) > 800: out = out[:800] + " …"
        lines.append(f"- {name}: {out}")
    return "\n".join(lines)

def _toolcalls_preview(tcs) -> str:
    try:
        out = []
        for tc in tcs:
            name = getattr(tc, "name", None) or getattr(getattr(tc, "tool", None), "name", None) or "tool"
            args = getattr(tc, "args", None)
            if args is None and hasattr(tc, "__dict__"):
                args = getattr(tc, "__dict__", {}).get("args", {})
            out.append({"name": name, "args": args or {}})
        return json.dumps(out, ensure_ascii=False)[:600]
    except Exception:
        try:
            out = [{"name": tc.get("name", "tool"), "args": tc.get("args", {})} for tc in tcs]
            return json.dumps(out, ensure_ascii=False)[:600]
        except Exception:
            return "(unserializable tool_calls)"

def _last_user_text(messages: List[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage) and (m.content or "").strip():
            return m.content
    return ""

# 선주입
def _recall_with_new_session(member_id: int, user_text: str) -> str:
    db2 = SessionLocal()
    try:
        return (recall_or_general_context(user_text, member_id, db2, top_k=3) or "").strip()
    finally:
        db2.close()

async def preload_context(state):
    member_id = state.get("member_id")
    if state.get("disable_preload"):
        state["base_system_text"] = state.get("base_system_text") or _ensure_role_text(member_id)
        state["preload_context"] = "없음"
        state["tool_context"] = state.get("tool_context", "") or "없음"
        state["tool_pass_done"] = False
        state["executed_tools"] = []
        return state

    messages: List[BaseMessage] = state.get("messages", []) or []
    user_text = _last_user_text(messages) or ""

    db = SessionLocal()
    try:
        summary_task = summarize_conversation(member_id, db, limit=20)
        recall_task = asyncio.to_thread(_recall_with_new_session, member_id, user_text)
        emotion_task = asyncio.to_thread(predict_emotion, user_text) if user_text else asyncio.to_thread(lambda: "")
        summary, recall_ctx, emotion = await asyncio.gather(summary_task, recall_task, emotion_task)
        if hasattr(summary, "content"): summary = summary.content
        summary = (summary or "").strip()
        recall_ctx = (recall_ctx or "").strip()
        emotion = (emotion or "").strip()
    finally:
        db.close()

    base_role = state.get("base_system_text") or _ensure_role_text(member_id)
    if summary:   base_role += f"\n\n[최근 대화 요약]\n{summary}"
    if recall_ctx:base_role += f"\n\n[회상 컨텍스트]\n{recall_ctx[:600]}"
    if emotion:   base_role += f"\n\n[사용자 현재 감정 추정] {emotion}"
    state["base_system_text"] = base_role

    add_lines = []
    if summary:    add_lines.append(f"- preload_summary: {summary[:400]}")
    if recall_ctx: add_lines.append(f"- preload_recall: {recall_ctx[:400]}")
    if emotion:    add_lines.append(f"- preload_emotion: {emotion}")
    state["preload_context"] = "\n".join(add_lines) or "없음"

    state["tool_context"] = state.get("tool_context", "") or "없음"
    state["tool_pass_done"] = False
    state["executed_tools"] = []
    return state

# 에이전트
async def call_agent(state):
    messages: List[BaseMessage] = state.get("messages", []) or []
    member_id = state.get("member_id")
    force_summary = bool(state.get("force_summary", False))

    if messages and isinstance(messages[-1], HumanMessage):
        state.setdefault("executed_tools", [])
        state["executed_tools"].clear()
        state["tool_pass_done"] = False

    if not state.get("base_system_text"):
        state["base_system_text"] = _ensure_role_text(member_id)

    history = _history_all(messages)
    tool_context = (state.get("tool_context") or "없음").strip()
    preload_context = (state.get("preload_context") or "없음").strip()

    has_preload = preload_context != "없음"
    control_hint = (
        "" if has_preload else (
            "최근 맥락을 반영하기 위해 summarize_tool(member_id, limit=20)을 우선 고려하라."
            if (force_summary or any(isinstance(m, AIMessage) for m in history)) else ""
        )
    )
    if state.get("tool_pass_done"):
        control_hint += ("\n[중요] 이미 도구를 1회 사용했다. 이번 턴에는 Final만 작성하라.")

    logger.info("=== REACT / THOUGHT === has_preload=%s tool_pass_done=%s history=%d preload_len=%d tool_ctx_len=%d",
                has_preload, state.get("tool_pass_done"), len(history), len(preload_context), len(tool_context))

    prompt_msgs = AGENT_PROMPT.format_messages(
        role_text=state["base_system_text"],
        control_hint=control_hint.strip(),
        history=history,
        preload_context=preload_context,
        tool_context=tool_context,
    )

    ai = await LLM_WITH_TOOLS.ainvoke(prompt_msgs)

    if getattr(ai, "tool_calls", None) and not state.get("tool_pass_done"):
        logger.info("=== REACT / DECISION === tool_calls -> %s", _toolcalls_preview(ai.tool_calls))
        return {"messages": messages + [ai]}

    preview = (ai.content or "").replace("\n", " ")
    if len(preview) > 120: preview = preview[:120] + " …"
    logger.info("=== REACT / DECISION === no_tool -> preview=\"%s\"", preview or "(empty)")
    return {"messages": messages + [ai]}

def should_call_tools(state) -> str:
    msgs: List[BaseMessage] = state.get("messages", [])
    last = msgs[-1] if msgs else None
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None) and not state.get("tool_pass_done"):
        logger.info("=== ROUTER === agent → tools")
        return "tools"
    logger.info("=== ROUTER === agent → finalize")
    return "finalize"

async def tools_to_prompt(state):
    tmsgs = _collect_recent_tool_msgs(state.get("messages", []))
    for tm in tmsgs:
        tname = getattr(tm, "name", "tool")
        content = (tm.content or "").strip()
        snippet = (content[:300] + " …") if len(content) > 300 else content
        logger.info("=== REACT / OBSERVATION === %s -> %s", tname, snippet)

    summary = _summarize_tools(tmsgs)
    executed = [getattr(tm, "name", "unknown") for tm in tmsgs]
    logger.info("=== REACT / OBS-SUMMARY === len=%d, executed=%s", len(summary), executed)

    prev = (state.get("tool_context") or "").strip()
    state["tool_context"] = (prev + ("\n" if prev and summary else "") + summary).strip() or "없음"

    state["executed_tools"] = executed
    state["tool_pass_done"] = True
    return state

async def finalize(state):
    resp = None
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage) and m.content:
            resp = m.content.strip(); break
    if not resp:
        resp = "음… 뭐라고 말해줄까 생각 중이야."
        logger.warning("=== REACT / FINAL === (empty → fallback)")
    else:
        logger.info("=== REACT / FINAL === %s", (resp[:200] + " …") if len(resp) > 200 else resp)
    return {"response": resp}
