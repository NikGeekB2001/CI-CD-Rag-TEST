import pytest
from src.medical_assistant import MedicalAssistant

@pytest.fixture(scope="module")
def assistant():
    # В реальном CI/CD можно использовать моки, чтобы не загружать модели
    # Для локального теста можно загрузить, если позволяет железо
    try:
        return MedicalAssistant()
    except Exception as e:
        pytest.skip(f"Не удалось загрузить модели: {e}")
