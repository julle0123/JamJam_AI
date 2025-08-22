# app/core/client.py
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from langchain_community.vectorstores import Qdrant as QdrantVectorStore
from app.core.config import settings

# --- OpenAI ---
os.environ["OPENAI_API_KEY"] = settings.openai_api_key

# --- LangSmith ---
if settings.langsmith_tracing:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    if settings.langsmith_endpoint:
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
    if settings.langsmith_project:
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project

# 기본 LLM
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.6).with_config({"run_name": "BaseLLM"})

# 에이전트용 LLM
def agent_llm():
    return ChatOpenAI(model="gpt-4.1-mini", temperature=0).with_config({"run_name": "AgentLLM"})

# --- Embedding / Qdrant ---
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
VECTOR_SIZE = 1536

qdrant_client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

def _ensure_collection_and_indexes():
    col = settings.collection_name
    try:
        cols = qdrant_client.get_collections().collections
        if not any(c.name == col for c in cols):
            qdrant_client.create_collection(
                collection_name=col,
                vectors_config=qmodels.VectorParams(size=VECTOR_SIZE, distance=qmodels.Distance.COSINE),
            )
    except Exception as e:
        print(f"[Qdrant] ensure collection warn: {e}")

    # ✅ 인덱스 스키마: member_id=KEYWORD(문자열) 로 통일
    for name, schema in [
        ("member_id", qmodels.PayloadSchemaType.KEYWORD),  # ← 변경: INTEGER -> KEYWORD
        ("role", qmodels.PayloadSchemaType.KEYWORD),
        ("created_at", qmodels.PayloadSchemaType.KEYWORD),
    ]:
        try:
            qdrant_client.create_payload_index(collection_name=col, field_name=name, field_schema=schema)
        except Exception as e:
            print(f"[Qdrant] index {name} warn: {e}")

_ensure_collection_and_indexes()

vectorstore = QdrantVectorStore(
    client=qdrant_client, collection_name=settings.collection_name, embeddings=embedding
)
