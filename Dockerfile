FROM python:3.10-slim

WORKDIR /app

# Копируем requirements и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код и модели
COPY . .

# Команда для запуска
CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0"]
