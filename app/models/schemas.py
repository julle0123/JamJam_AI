# app/models/schemas.py
from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional

# 챗봇 대화 요청 입력 스키마
class ChatRequest(BaseModel):
    member_id: int               # DB member_id와 매핑
    input: str                   # 사용자 입력 문장
    session_id: Optional[str] = None
    force_summary: Optional[bool] = False
    disable_preload: Optional[bool] = False  # 선주입(요약/회상/감정) 끄기
    debug_trace: Optional[bool] = False      # 실시간 LLM 스트림/툴콜 트레이스

# 챗봇 대화 응답 스키마
class ChatResponse(BaseModel):
    output: str
    user_emotion: Optional[str] = None

# 대화 로그 저장용 입력 스키마
class ChatLogCreate(BaseModel):
    member_id: int
    user_text: str
    bot_text: str

# 대화 로그 조회용 출력 스키마
class ChatLogResponse(ChatLogCreate):
    chat_id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
