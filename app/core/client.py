# app/core/client.py
# OpenAI LLM/임베딩, Qdrant 클라이언트 및 벡터스토어 초기화.
import os
import logging
from typing import Dict, Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from langchain_community.vectorstores import Qdrant as QdrantVectorStore
from app.core.config import settings

log = logging.getLogger("infra.qdrant")

# --- OpenAI ---
os.environ["OPENAI_API_KEY"] = settings.openai_api_key

# --- LangSmith (선택) ---
if settings.langsmith_tracing:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    if settings.langsmith_endpoint:
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
    if settings.langsmith_project:
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project

# 기본 LLM (요약 등에 사용) — 스트리밍 ON
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.6,
    timeout=30,
    streaming=True,                 # ← 스트리밍 활성화
).with_config({"run_name": "BaseLLM"})

# 에이전트용 LLM — 스트리밍 ON
def agent_llm():
    return ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.4,
        timeout=30,
        streaming=True,             # ← 스트리밍 활성화
    ).with_config({"run_name": "AgentLLM"})

# --- Embedding / Qdrant ---
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
VECTOR_SIZE = 1536

qdrant_client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

def _payload_schema(name: str) -> Optional[qmodels.PayloadSchemaType]:
    try:
        info = qdrant_client.get_collection(settings.collection_name)
        schema: Dict[str, qmodels.PayloadSchemaInfo] = getattr(info, "payload_schema", {}) or {}
        if name in schema and getattr(schema[name], "data_type", None):
            return schema[name].data_type
    except Exception as e:
        log.debug(f"[Qdrant] read schema warn: {e}")
    return None

def _ensure_collection_and_indexes():
    col = settings.collection_name
    try:
        cols = qdrant_client.get_collections().collections
        if not any(c.name == col for c in cols):
            qdrant_client.create_collection(
                collection_name=col,
                vectors_config=qmodels.VectorParams(size=VECTOR_SIZE, distance=qmodels.Distance.COSINE),
            )
            log.info(f"[Qdrant] created collection: {col}")
    except Exception as e:
        log.warning(f"[Qdrant] ensure collection warn: {e}")

    for name, schema in [
        ("member_id", qmodels.PayloadSchemaType.KEYWORD),
        ("role", qmodels.PayloadSchemaType.KEYWORD),
        ("created_at", qmodels.PayloadSchemaType.KEYWORD),
    ]:
        try:
            cur = _payload_schema(name)
            if cur != schema:
                try:
                    qdrant_client.delete_payload_index(collection_name=col, field_name=name)
                except Exception:
                    pass
                qdrant_client.create_payload_index(
                    collection_name=col,
                    field_name=name,
                    field_schema=schema,
                    wait=True,       # ← 인덱싱 완료까지 대기
                )
                log.info(f"[Qdrant] ensured index: {name} -> {schema}")
        except Exception as e:
            log.warning(f"[Qdrant] index ensure warn for {name}: {e}")

_ensure_collection_and_indexes()

vectorstore = QdrantVectorStore(
    client=qdrant_client,
    collection_name=settings.collection_name,
    embeddings=embedding,
)
