import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from medical_assistant import MedicalAssistant

st.set_page_config(page_title="Медицинский Ассистент", layout="wide")

st.markdown("""
<style>
    /* Ваш CSS код */
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assistant():
    """Загружаем ассистента один раз и кэшируем"""
    return MedicalAssistant()

st.title("🩺 AI-Медицинский Ассистент")
st.caption("Ваши вопросы - моя база знаний. Помните: это не замена консультации врача.")

# Инициализация ассистента
assistant = load_assistant()

if not assistant.models_loaded:
    st.error("Не удалось загрузить модели. Проверьте консоль и доступные ресурсы (GPU/ОЗУ).")
    st.stop()

# Поле для ввода вопроса
user_question = st.text_input(
    "Задайте ваш вопрос:",
    placeholder="Например: 'Что делать при температуре 38.5?' или 'Какую мазь использовать от ушиба?'"
)

if user_question:
    with st.spinner("Анализирую вопрос и ищу информацию..."):
        response = assistant.process_question(user_question)
    
    st.markdown("---")
    st.markdown(response, unsafe_allow_html=True)
