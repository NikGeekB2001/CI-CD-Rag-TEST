# RAG Evaluator for medical assistant

import re
from transformers import pipeline

class RAGEvaluator:
    def __init__(self):
        # Load local models for evaluation
        self.qa_pipeline = pipeline("question-answering", model="Den4ikAI/rubert_large_squad_2")

    def evaluate(self, question, context, answer):
        """Evaluate RAG response using local models"""
        try:
            # Faithfulness: Check if answer is supported by context
            faithfulness = self._calculate_faithfulness(answer, context)

            # Relevance: Check if answer addresses the question
            relevance = self._calculate_relevance(question, answer)

            # Correctness: Simple check based on medical keywords
            correctness = self._calculate_correctness(answer)

            return {
                "faithfulness": faithfulness,
                "relevance": relevance,
                "correctness": correctness
            }
        except Exception as e:
            print(f"Error in evaluation: {e}")
            return {
                "faithfulness": 0.0,
                "relevance": 0.0,
                "correctness": 0.0
            }

    def _calculate_faithfulness(self, answer, context):
        """Check if answer is faithful to context"""
        if not context or not answer:
            return 0.0

        # Simple word overlap
        answer_words = set(re.findall(r'\b\w+\b', answer.lower()))
        context_words = set(re.findall(r'\b\w+\b', context.lower()))

        overlap = len(answer_words.intersection(context_words))
        total = len(answer_words)

        if total == 0:
            return 0.0

        return min(overlap / total, 1.0)

    def _calculate_relevance(self, question, answer):
        """Check if answer is relevant to question"""
        if not question or not answer:
            return 0.0

        try:
            # Use QA model to extract answer from the generated answer
            result = self.qa_pipeline(question=question, context=answer)
            confidence = result['score'] if isinstance(result, dict) and 'score' in result else 0.0
            return confidence
        except Exception as e:
            print(f"QA pipeline error: {e}")
            # Fallback: keyword matching
            question_words = set(re.findall(r'\b\w+\b', question.lower()))
            answer_words = set(re.findall(r'\b\w+\b', answer.lower()))

            overlap = len(question_words.intersection(answer_words))
            return min(overlap / len(question_words), 1.0) if question_words else 0.0

    def _calculate_correctness(self, answer):
        """Simple correctness check based on medical content"""
        medical_keywords = [
            'врач', 'лечение', 'симптом', 'болезнь', 'здоровье', 'медицин',
            'препарат', 'таблетка', 'мазь', 'температура', 'боль', 'рана'
        ]

        answer_lower = answer.lower()
        matches = sum(1 for keyword in medical_keywords if keyword in answer_lower)

        # Check for disclaimers
        has_disclaimer = any(phrase in answer_lower for phrase in [
            'обратитесь к врачу', 'консультация врача', 'не заменяет'
        ])

        base_score = min(matches / 3, 1.0)  # At least 3 keywords for full score
        disclaimer_bonus = 0.2 if has_disclaimer else 0.0

        return min(base_score + disclaimer_bonus, 1.0)
