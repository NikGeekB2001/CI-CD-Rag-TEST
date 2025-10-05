import json
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
from sentence_transformers import SentenceTransformer

import time
from pymilvus import exceptions

def init_milvus():
    # Retry connection to Milvus until successful or timeout
    max_retries = 10
    retry_delay = 5  # seconds
    for attempt in range(max_retries):
        try:
            connections.connect(alias="default", host='milvus-standalone', port='19530')
            print("Connected to Milvus")
            break
        except exceptions.MilvusException as e:
            print(f"Connection attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(retry_delay)

    # Загрузка данных
    with open('data/medical_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Модель для эмбеддингов
    model = SentenceTransformer("intfloat/multilingual-e5-large")

    # Создание схемы коллекции
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
    ]
    schema = CollectionSchema(fields, "Medical knowledge base")

    # Создание коллекции
    collection_name = "medical_collection"
    try:
        collection = Collection(collection_name)
        print(f"Коллекция '{collection_name}' уже существует.")
    except:
        collection = Collection(collection_name, schema)
        print(f"Коллекция '{collection_name}' создана.")

    # Подготовка данных
    ids = []
    embeddings = []
    texts = []

    for item in data:
        text = f"{item['question']} {item['answer']}"
        embedding = model.encode(f"query: {text}")
        if hasattr(embedding, 'tolist'):
            embedding = embedding.tolist()

        ids.append(item['id'])
        embeddings.append(embedding)
        texts.append(text)

    # Вставка данных
    entities = [ids, embeddings, texts]
    collection.insert(entities)
    collection.flush()

    # Создание индекса
    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index("embedding", index_params)

    # Загрузка коллекции
    collection.load()

    print(f"✅ Инициализация завершена. Вставлено {len(data)} записей.")

if __name__ == "__main__":
    init_milvus()
