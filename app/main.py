# app/main.py
# FastAPI 앱 부트스트랩, 로깅 수준 일괄 조정(소음 억제), 라우터/테이블 초기화.

from fastapi import FastAPI
from app.api import chat
from app.models.base import Base
from app.core.db import engine
from dotenv import load_dotenv
import app.models

import logging
from logging import StreamHandler, Formatter

load_dotenv()

def configure_logging():
    root = logging.getLogger()
    if not root.handlers:
        h = StreamHandler()
        h.setFormatter(Formatter("[%(asctime)s][%(levelname)s] %(name)s: %(message)s"))
        root.addHandler(h)
    root.setLevel(logging.INFO)

    # 에이전트 로거
    logging.getLogger("react").setLevel(logging.INFO)

    # SQLAlchemy 상세 로그 억제
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.orm").setLevel(logging.WARNING)
    logging.getLogger("alembic").setLevel(logging.WARNING)

    # 외부 HTTP 클라이언트 소음 억제(선택)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

configure_logging()

app = FastAPI(title="JAMJAM AI")

# 테이블 생성(프로덕션은 마이그레이션 권장)
Base.metadata.create_all(bind=engine)

# 라우터
app.include_router(chat.router, prefix="/chat", tags=["Chat"])

@app.get("/")
def root():
    return {"message": "JAMJAM AI"}

