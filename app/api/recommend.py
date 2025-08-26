import os
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from app.core.client import qdrant_client, embedding
from qdrant_client.http import models as qmodels

router = APIRouter()

# --- 정책 추천 전용 컬렉션 이름 ---
COLLECTION_NAME = os.getenv("COLLECTION_NAME2", "policy_embeddings")

# --- 요청/응답 모델 ---
class RecommendRequest(BaseModel):
    region: str
    current_status: List[str]
    childbirth_status: Optional[int] = 0  # 0=무관, 1=출산, 2=임신
    marriage_status: Optional[int] = 0    # 0=무관, 1=기혼, 2=결혼예정
    children_count: Optional[int] = None
    income: Optional[int] = None

class RecommendResponse(BaseModel):
    policy_id: int
    title: str

# --- 인덱스 보장 함수 ---
def ensure_policy_indexes():
    fields_to_index = [
        ("region", qmodels.PayloadSchemaType.KEYWORD),
        ("childbirth_status", qmodels.PayloadSchemaType.INTEGER),
        ("marriage_status", qmodels.PayloadSchemaType.INTEGER),
    ]
    for name, schema in fields_to_index:
        try:
            qdrant_client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=name,
                field_schema=schema,
                wait=True,
            )
        except Exception:
            # 이미 있으면 무시
            pass

# API 초기화 시 인덱스 보장
ensure_policy_indexes()

# --- 쿼리 텍스트 생성 ---
def build_query_text(req: RecommendRequest) -> str:
    conds = [f"지역={req.region}", f"상태={','.join(req.current_status)}"]
    if req.childbirth_status == 1:
        conds.append("출산가정")
    elif req.childbirth_status == 2:
        conds.append("임산부")
    if req.marriage_status == 1:
        conds.append("기혼")
    elif req.marriage_status == 2:
        conds.append("결혼예정")
    if req.children_count is not None:
        conds.append(f"자녀수={req.children_count}")
    if req.income is not None:
        conds.append(f"중위소득={req.income}%")
    return " ".join(conds)

# --- 정책 추천 API ---
@router.post("/recommend", response_model=List[RecommendResponse])
def recommend(req: RecommendRequest):
    # 1. 사용자 입력 임베딩
    query_text = build_query_text(req)
    query_vector = embedding.embed_query(query_text)

    # 2. 필터 조건
    filters = []
    if req.region:
        filters.append(qmodels.FieldCondition(
            key="region", match=qmodels.MatchAny(any=[req.region])
        ))
    if req.childbirth_status:
        filters.append(qmodels.FieldCondition(
            key="childbirth_status", match=qmodels.MatchValue(value=req.childbirth_status)
        ))
    if req.marriage_status:
        filters.append(qmodels.FieldCondition(
            key="marriage_status", match=qmodels.MatchValue(value=req.marriage_status)
        ))

    query_filter = qmodels.Filter(must=filters) if filters else None

    # 3. Qdrant 검색
    results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=5,
        query_filter=query_filter,
    )

    # 4. 응답 변환
    return [
        RecommendResponse(policy_id=r.payload["policy_id"], title=r.payload["title"])
        for r in results
    ]
