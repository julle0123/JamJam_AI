# app/core/db.py
# SQLAlchemy 엔진/세션팩토리. 여기서 로그 소음 억제(echo=False).
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

# echo=False로 SQL 로그 억제(필요 시 .env의 sqlalchemy_echo=true로만 활성)
engine = create_engine(
    settings.database_url,
    echo=settings.sqlalchemy_echo,
    pool_pre_ping=True,         # 죽은 커넥션 자동 감지
    pool_recycle=1800,          # 장시간 유휴 연결 재활용
    future=True
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
