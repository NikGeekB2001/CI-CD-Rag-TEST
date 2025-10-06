import pytest
import os
from src.medical_assistant import MedicalAssistant

class MockMedicalAssistant:
    def answer_question(self, question):
        return f"Mock answer for: {question}"

    def generate_response(self, query):
        return f"Mock response for: {query}"

    def get_question_type(self, question):
        if "привет" in question.lower():
            return "приветствие"
        elif "мазь" in question.lower():
            return "мазь"
        elif "температура" in question.lower():
            return "температура"
        else:
            return "объяснение"

    def generate_template_response(self, question, context, entities, question_type):
        return "### Медицинская консультация\n\nMock response for template. парацетамол к врачу"

    # Add other methods as needed to match MedicalAssistant interface

class MockRAGPipeline:
    def search(self, query):
        return f"Mock context for: {query}"

    def generate_answer(self, query, context):
        return f"Mock answer for: {query} with context: {context}"

@pytest.fixture(scope="module")
def assistant():
    if os.getenv('CI'):
        return MockMedicalAssistant()
    else:
        try:
            return MedicalAssistant()
        except Exception as e:
            pytest.skip(f"Не удалось загрузить модели: {e}")

@pytest.fixture(scope="module")
def rag_pipeline():
    if os.getenv('CI'):
        return MockRAGPipeline()
    else:
        from src.rag_pipeline import RAGPipeline
        try:
            return RAGPipeline()
        except Exception as e:
            pytest.skip(f"Не удалось загрузить RAG pipeline: {e}")
