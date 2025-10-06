import pytest
import os
from src.medical_assistant import MedicalAssistant

@pytest.fixture(scope="module")
def assistant():
    try:
        return MedicalAssistant()
    except Exception as e:
        pytest.skip(f"Не удалось загрузить модели: {e}")

@pytest.fixture(scope="module")
def rag_pipeline():
    from src.rag_pipeline import RAGPipeline
    try:
        return RAGPipeline()
    except Exception as e:
        pytest.skip(f"Не удалось загрузить RAG pipeline: {e}")
