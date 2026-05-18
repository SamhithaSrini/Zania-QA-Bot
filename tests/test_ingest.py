from unittest.mock import MagicMock, patch

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import ContextualCompressionRetriever

from app.core.config import settings
from app.services.ingest import build_retriever


class FakeRetriever(BaseRetriever):
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        return []


class FakeCompressor(BaseDocumentCompressor):
    def compress_documents(
        self,
        documents: list[Document],
        query: str,
        callbacks=None,
    ) -> list[Document]:
        return documents


@patch("app.services.ingest.CrossEncoderReranker")
@patch("app.services.ingest.HuggingFaceCrossEncoder")
@patch("app.services.ingest.FAISS.from_documents")
@patch("app.services.ingest.OpenAIEmbeddings")
def test_build_retriever_returns_base_retriever_when_reranker_disabled(
    mock_embeddings,
    mock_from_documents,
    mock_cross_encoder,
    mock_reranker,
) -> None:
    settings.USE_RERANKER = False
    vector_store = MagicMock()
    base_retriever = FakeRetriever()
    vector_store.as_retriever.return_value = base_retriever
    mock_from_documents.return_value = vector_store

    retriever = build_retriever("The sky is blue.")

    assert retriever is base_retriever
    mock_embeddings.assert_called_once()
    mock_from_documents.assert_called_once()
    vector_store.as_retriever.assert_called_once_with(search_kwargs={"k": 5})
    mock_cross_encoder.assert_not_called()
    mock_reranker.assert_not_called()


@patch("app.services.ingest.CrossEncoderReranker")
@patch("app.services.ingest.HuggingFaceCrossEncoder")
@patch("app.services.ingest.FAISS.from_documents")
@patch("app.services.ingest.OpenAIEmbeddings")
def test_build_retriever_returns_contextual_compression_retriever_when_enabled(
    mock_embeddings,
    mock_from_documents,
    mock_cross_encoder,
    mock_reranker,
) -> None:
    settings.USE_RERANKER = True
    vector_store = MagicMock()
    base_retriever = FakeRetriever()
    vector_store.as_retriever.return_value = base_retriever
    mock_from_documents.return_value = vector_store
    mock_reranker.return_value = FakeCompressor()

    retriever = build_retriever("The sky is blue.")

    assert isinstance(retriever, ContextualCompressionRetriever)
    mock_embeddings.assert_called_once()
    mock_from_documents.assert_called_once()
    vector_store.as_retriever.assert_called_once_with(search_kwargs={"k": 5})
    mock_cross_encoder.assert_called_once()
    mock_reranker.assert_called_once()
    settings.USE_RERANKER = False
