import json
import os
import pandas as pd
from evaluation.ragasevaluator import RAGEvaluator
from src.rag_pipeline import RAGPipeline

def create_evaluation_dataset():
    """Создание датасета для оценки RAG из медицинских данных"""
    with open('data/medical_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Создаем датасет для оценки
    eval_data = []
    for item in data[:10]:  # Берем первые 10 записей для теста
        question = item['question']
        ground_truth = item['answer']

        # Используем RAG для получения контекста
        rag = RAGPipeline()
        context = rag.search(question)

        eval_data.append({
            'question': question,
            'contexts': [context] if context else [""],
            'ground_truth': ground_truth,
            'answer': context  # Для простоты используем контекст как ответ
        })

    return Dataset.from_list(eval_data)

def evaluate_rag():
    """Оценка RAG-конвейера с помощью локального оценщика"""
    print("Создание датасета для оценки...")
    with open('data/medical_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("Запуск оценки RAG с помощью локального оценщика...")

    evaluator = RAGEvaluator()
    results = []

    for i, item in enumerate(data[:10]):  # Берем первые 10 записей для теста
        question = item['question']
        ground_truth = item['answer']

        # Используем RAG для получения контекста
        rag = RAGPipeline()
        context = rag.search(question)

        if not context:
            context = ""

        # Оценка
        scores = evaluator.evaluate(question, context, ground_truth)

        results.append({
            'question': question,
            'context': context,
            'ground_truth': ground_truth,
            'faithfulness': scores['faithfulness'],
            'relevance': scores['relevance'],
            'correctness': scores['correctness']
        })

        print(f"Обработано {i+1}/10: faithfulness={scores['faithfulness']:.3f}, relevance={scores['relevance']:.3f}, correctness={scores['correctness']:.3f}")

    # Средние значения
    avg_faithfulness = sum(r['faithfulness'] for r in results) / len(results)
    avg_relevance = sum(r['relevance'] for r in results) / len(results)
    avg_correctness = sum(r['correctness'] for r in results) / len(results)

    print("\nСредние результаты оценки:")
    print(f"Faithfulness: {avg_faithfulness:.3f}")
    print(f"Relevance: {avg_relevance:.3f}")
    print(f"Correctness: {avg_correctness:.3f}")

    # Сохранение результатов
    df = pd.DataFrame(results)
    df.to_csv('evaluation/ragas_results.csv', index=False)
    print("Результаты сохранены в evaluation/ragas_results.csv")

    return {
        'faithfulness': avg_faithfulness,
        'relevance': avg_relevance,
        'correctness': avg_correctness
    }

if __name__ == "__main__":
    evaluate_rag()
