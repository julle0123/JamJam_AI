import json
from app.graph.prompts import load_prompt_template
from app.core.client import llm
from app.graph.tools import classify_emotion, rag_search, summarize

# 1. 툴 선택 노드
def decide_tools_node(state):
    prompt = load_prompt_template("tool_decision").format(user_input=state["user_input"])
    decision = llm.invoke(prompt)
    try:
        state["tool_flags"] = json.loads(decision)
    except:
        state["tool_flags"] = {"use_emotion": False, "use_rag": False, "use_summary": False}
    return state

# 2. 감정분류
def classify_emotion_node(state):
    state["emotion"] = classify_emotion(state["user_input"])
    return state

# 3. RAG 검색
def rag_node(state):
    state["memory"] = rag_search(state["user_input"])
    return state

# 4. 전체 대화 요약
def summary_node(state):
    state["summary"] = summarize(state["member_id"])
    return state

# 5. 최종 응답 생성
def generate_response_node(state):
    history_text = "\n".join(
        [f"{m.type.upper()}: {m.content}" for m in state.get("history", [])]
    )

    prompt = load_prompt_template("response").format(
        user_input=state["user_input"],
        emotion=state.get("emotion", ""),
        memory=state.get("memory", ""),
        summary=state.get("summary", ""),
        history=history_text
    )
    state["response"] = llm.invoke(prompt)
    return state
