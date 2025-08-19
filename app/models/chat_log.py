from sqlalchemy import Column, BigInteger, DateTime, Text, ForeignKey
from datetime import datetime
from app.models.base import Base
from sqlalchemy.orm import relationship

class ChatLog(Base):
    __tablename__ = "chat_log"

    chat_id = Column(BigInteger, primary_key=True, autoincrement=True)
    member_id = Column(BigInteger, ForeignKey("member.member_id", ondelete="CASCADE", onupdate="RESTRICT"), nullable=False)
    user_text = Column(Text, nullable=False)
    bot_text = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # 관계 매핑 (User 모델과 연결)
    member = relationship("User", backref="chat_logs")

    def __repr__(self):
        return f"<ChatLog(chat_id={self.chat_id}, member_id={self.member_id}, created_at={self.created_at})>"
