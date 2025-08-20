from fastapi import FastAPI
from app.api import chat   # chat.py 라우터 import
from app.models.base import Base            # SQLAlchemy Base 모델
from app.core.db import engine              # DB 연결 유지

app = FastAPI(title="JAMJAM AI")

Base.metadata.create_all(bind=engine)

# 라우터 등록
app.include_router(chat.router, prefix="/chat", tags=["Chat"])


# 기본 루트 엔드포인트
@app.get("/")
def root():
    return {"message": "JAMJAM AI"}
