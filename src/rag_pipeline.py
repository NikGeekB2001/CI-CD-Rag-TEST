from pymilvus import Collection, connections
from sentence_transformers import SentenceTransformer
import json

class RAGPipeline:
    def __init__(self, collection_name="medical_collection", model_name="intfloat/multilingual-e5-large"):
        self.model_name = model_name
        self.embedding_model = SentenceTransformer(model_name)
        self.collection_name = collection_name
        self.collection = None
        self._connect_to_milvus()

    def _connect_to_milvus(self):
        try:
            connections.connect(alias="default", host='localhost', port='19530')
            self.collection = Collection(self.collection_name)
            self.collection.load()
            print(f"✅ Успешное подключение к Milvus, коллекция '{self.collection_name}' загружена.")
        except Exception as e:
            print(f"❌ Ошибка подключения к Milvus: {e}")
            self.collection = None

    def search(self, query: str, top_k: int = 3) -> str:
        if not self.collection:
            return "Ошибка: База знаний (Milvus) недоступна."

        # E5-large требует префикса "query: "
        formatted_query = f"query: {query}"
        query_embedding = self.embedding_model.encode(formatted_query).tolist()
        
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text"]
        )
        
        if not results[0]:
            return "Информация по запросу не найдена в базе знаний."
            
        retrieved_contexts = [hit.entity.get("text") for hit in results[0]]
        return "\n\n".join(retrieved_contexts)
