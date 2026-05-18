from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OPENAI_API_KEY: str = ""
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    FAISS_TOP_K: int = 5
    RERANKER_TOP_N: int = 5
    USE_RERANKER: bool = False
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "gpt-4o-mini"
    MAX_DOCUMENT_BYTES: int = 25 * 1024 * 1024
    MAX_QUESTIONS: int = 25
    MAX_QUESTION_CHARS: int = 1000
    MAX_CONCURRENT_QUESTIONS: int = 5
    ENABLE_RAG_SCORES: bool = True
    MAX_CONCURRENT_JUDGE_CALLS: int = 3
    REQUEST_TIMEOUT_SECONDS: int = 120

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
