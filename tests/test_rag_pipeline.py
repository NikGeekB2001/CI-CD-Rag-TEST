import pytest
from src.rag_pipeline import RAGPipeline

def test_rag_pipeline_init():
    rag = RAGPipeline()
    assert rag.collection_name == "medical_collection"
    assert rag.model_name == "intfloat/multilingual-e5-large"

def test_rag_pipeline_search():
    rag = RAGPipeline()
    if rag.collection:  # Тест только если Milvus доступен
        result = rag.search("простуда")
        assert isinstance(result, str)
        assert len(result) > 0
    else:
        result = rag.search("простуда")
        assert "Ошибка: База знаний (Milvus) недоступна." in result
