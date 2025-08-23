# app/models/chat_log.py
# 대화 로그 테이블 모델. member와 FK 관계.
from sqlalchemy import Column, BigInteger, DateTime, Text, ForeignKey
from datetime import datetime
from sqlalchemy.orm import relationship
from app.models.base import Base

class ChatLog(Base):
    __tablename__ = "chat_log"

    chat_id = Column(BigInteger, primary_key=True, autoincrement=True)
    member_id = Column(BigInteger, ForeignKey("member.member_id", ondelete="CASCADE", onupdate="RESTRICT"), nullable=False, index=True)
    user_text = Column(Text, nullable=False)
    bot_text = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # 역참조: member.chat_logs
    member = relationship("User", back_populates="chat_logs")

    def __repr__(self):
        return f"<ChatLog(chat_id={self.chat_id}, member_id={self.member_id}, created_at={self.created_at})>"


