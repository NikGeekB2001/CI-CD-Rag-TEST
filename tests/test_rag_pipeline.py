import pytest
import os

def test_rag_pipeline_init(rag_pipeline):
    if os.getenv('CI'):
        # In CI, mock doesn't have these attributes
        pytest.skip("Skip init test in CI")
    else:
        assert rag_pipeline.collection_name == "medical_collection"
        assert rag_pipeline.model_name == "intfloat/multilingual-e5-large"

def test_rag_pipeline_search(rag_pipeline):
    result = rag_pipeline.search("простуда")
    if os.getenv('CI'):
        assert "Mock context for: простуда" in result
    else:
        if hasattr(rag_pipeline, 'collection') and rag_pipeline.collection:  # Тест только если Milvus доступен
            assert isinstance(result, str)
            assert len(result) > 0
        else:
            assert "Ошибка: База знаний (Milvus) недоступна." in result
