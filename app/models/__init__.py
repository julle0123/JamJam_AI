# app/models/__init__.py
# 모델 모듈 import를 통해 Base.metadata에 테이블 등록을 보장한다.
from .user import User  # noqa: F401
from .chat_log import ChatLog  # noqa: F401
