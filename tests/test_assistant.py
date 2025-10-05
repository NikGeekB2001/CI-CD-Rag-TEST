import pytest
from src.medical_assistant import MedicalAssistant

def test_get_question_type(assistant):
    assert assistant.get_question_type("Привет") == "приветствие"
    assert assistant.get_question_type("Какую мазь от синяка?") == "мазь"
    assert assistant.get_question_type("У меня температура 39") == "температура"
    assert assistant.get_question_type("Что такое гипертония?") == "объяснение"

def test_template_generation(assistant):
    question = "Что делать при температуре?"
    context = "Пейте много жидкости."
    entities = "Найденные сущности: температура (SYMPTOM)"
    
    response = assistant.generate_template_response(question, context, entities, "температура")
    
    assert "### Медицинская консультация" in response
    assert "парацетамол" in response.lower()
    assert "к врачу" in response

# Тест для RAG-конвейера
def test_rag_pipeline_search():
    from src.rag_pipeline import RAGPipeline
    rag = RAGPipeline()
    if rag.collection: # Тест только если Milvus доступен
        result = rag.search("простуда")
        assert isinstance(result, str)
        assert len(result) > 0
