from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings

from app.core.config import settings

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def build_retriever(text: str) -> BaseRetriever:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    chunks = splitter.create_documents([text])
    embeddings = OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY,
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    base_retriever = vector_store.as_retriever(search_kwargs={"k": settings.FAISS_TOP_K})

    if not settings.USE_RERANKER:
        return base_retriever

    cross_encoder = HuggingFaceCrossEncoder(model_name=CROSS_ENCODER_MODEL)
    reranker = CrossEncoderReranker(
        model=cross_encoder,
        top_n=settings.RERANKER_TOP_N,
    )
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever,
    )
