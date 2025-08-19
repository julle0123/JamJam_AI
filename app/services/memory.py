from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from typing import Dict
from app.core.client import vectorstore

_store: Dict[str, ChatMessageHistory] = {}

def get_user_history(user_id: str) -> BaseChatMessageHistory:
    if user_id not in _store:
        _store[user_id] = ChatMessageHistory()
    return _store[user_id]

def search_memory(query: str, top_k: int = 3) -> str:
    results = vectorstore.similarity_search(query, k=top_k)
    if not results:
        return ""
    return "\n".join([doc.page_content for doc in results])
