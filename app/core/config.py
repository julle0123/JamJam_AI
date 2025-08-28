# app/core/config.py
# 환경변수 기반 설정. DB URL 조합 포함.
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str

    # Qdrant
    qdrant_url: str
    qdrant_api_key: str
    collection_name: str = "jamjam_history"
    collection_name2 :str = "policy_embeddings"

    # LangSmith
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_endpoint: str | None = None
    langsmith_project: str | None = None

    # MySQL
    mysql_host: str
    mysql_port: int
    mysql_user: str
    mysql_password: str
    mysql_db: str

    # Logging 옵션 (.env로 제어 가능)
    sqlalchemy_echo: bool = False                 # SQL 원문 로깅(운영 기본 꺼짐)
    sqlalchemy_log_level: str = "WARNING"         # sqlalchemy.engine 로거 레벨
    react_log_level: str = "INFO"                 # ReAct 로거 레벨(app.main에서 적용)

    # Supertone TTS
    SUPERTONE_API_KEY: str
    SUPERTONE_TTS_ENDPOINT: str = "https://api.supertone.ai/tts"
    AUDIO_CACHE_DIR: str = "data/audio"
    SUPERTONE_TOBY_VOICE_ID: str
    SUPERTONE_EMOTION_PARAM_NAME: str = "style"
    
    @property
    def database_url(self) -> str:
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_db}?charset=utf8mb4"
        )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

settings = Settings()
