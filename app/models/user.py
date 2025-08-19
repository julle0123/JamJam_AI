from sqlalchemy import Column, String, DateTime, BigInteger, Integer
from datetime import datetime
from app.models.base import Base  

class User(Base):
    __tablename__ = "member"

    member_id = Column(BigInteger, primary_key=True, autoincrement=True)
    provider = Column(Integer, nullable=False)  # 1=KAKAO, 2=GOOGLE
    provider_user_id = Column(String(191), nullable=False)
    nickname = Column(String(50), nullable=True)
    email = Column(String(191), nullable=True)
    profile_image_url = Column(String(255), nullable=True)
    joined_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_login_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<User(member_id={self.member_id}, provider={self.provider}, nickname={self.nickname})>"
