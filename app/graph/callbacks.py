# app/graph/callbacks.py
import json
import logging
from typing import Any, Dict, Optional
from langchain_core.callbacks import BaseCallbackHandler

class ReactTraceCallback(BaseCallbackHandler):
    """
    LLM 스트림/툴콜 델타를 터미널에 그대로 출력.
    - content 토큰: 개행 이스케이프, 120자까지만 미리보기
    - tool_calls 델타: 함수명과 arguments 누적 프리뷰
    - Tool 실행 전/후의 input/output 스니펫
    """
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.log = logger or logging.getLogger("react")

    # LLM
    def on_chat_model_start(self, serialized: Dict[str, Any], messages, **kwargs):
        name = serialized.get("name") or serialized.get("id") or "chat_model"
        self.log.info("[TRACE] LLM start: %s", name)

    def on_chat_model_stream(self, chunk, **kwargs):
        content = getattr(chunk, "content", None)
        if content:
            s = content.replace("\n", "\\n")
            if len(s) > 120:
                s = s[:120] + " …"
            self.log.info("[TRACE] LLM> %s", s)

        tool_calls = getattr(chunk, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                name = getattr(tc, "name", None)
                fn = getattr(tc, "function", None)
                if fn is not None:
                    name = name or getattr(fn, "name", None)
                    args = getattr(fn, "arguments", None)
                else:
                    args = None
                if args is not None:
                    preview = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
                    if len(preview) > 120:
                        preview = preview[:120] + " …"
                    self.log.info("[TRACE] tool_call(delta): %s args+=%s", name, preview)

    def on_chat_model_end(self, outputs, **kwargs):
        self.log.info("[TRACE] LLM end")

    # Tool 실행
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs):
        name = serialized.get("name") or "tool"
        prev = input_str if len(input_str) <= 200 else input_str[:200] + " …"
        self.log.info("[TRACE] Tool start: %s input=%s", name, prev)

    def on_tool_end(self, output: str, **kwargs):
        out = output if output is not None else ""
        if len(out) > 200:
            out = out[:200] + " …"
        self.log.info("[TRACE] Tool end: %s", out)
