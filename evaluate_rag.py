"""
Скрипт для оценки качества RAG-системы через RAGAS
"""
import os
import math
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from datasets import Dataset
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from rag_assistant import ask_assistant
import config

EVALUATION_QUESTIONS = [
    "Какие правила работы сервисной службы?",
    "Как восстановить доступ к аккаунту?",
    "Какое время ответа на обращение клиента?",
    "Можно ли использовать продукт на нескольких устройствах?",
    "Как экспортировать данные из системы?"
]


def prepare_dataset(questions: list[str]) -> Dataset:
    """
    Подготовка датасета для RAGAS из вопросов

    Args:
        questions: список вопросов

    Returns:
        Dataset для RAGAS
    """
    questions_list = []
    answers_list = []
    contexts_list = []
    ground_truths_list = []

    print("Получение ответов от ассистента...")

    for i, question in enumerate(questions, 1):
        print(f"  Обработка вопроса {i}/{len(questions)}: {question}")

        result = ask_assistant(question)

        questions_list.append(question)
        answers_list.append(result["answer"])

        context_texts = [chunk["document"] for chunk in result["context"]]
        contexts_list.append(context_texts)

        ground_truths_list.append("")

    dataset_dict = {
        "question": questions_list,
        "answer": answers_list,
        "contexts": contexts_list,
        "ground_truth": ground_truths_list
    }

    dataset = Dataset.from_dict(dataset_dict)
    return dataset


def evaluate_rag_system():
    """
    Основная функция оценки RAG-системы
    """
    print("=" * 60)
    print("Оценка качества RAG-системы через RAGAS")
    print("=" * 60)

    dataset = prepare_dataset(EVALUATION_QUESTIONS)

    print("\nЗапуск оценки метрик...")
    print("Метрики: faithfulness, answer_relevancy, context_precision")

    os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

    langchain_embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=config.OPENAI_API_KEY
    )
    langchain_llm = ChatOpenAI(
        model_name=config.CHAT_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
        temperature=0
    )

    ragas_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)
    ragas_llm = LangchainLLMWrapper(langchain_llm)

    faithfulness_metric = Faithfulness(llm=ragas_llm)
    answer_relevancy_metric = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)

    try:
        context_precision_metric = ContextPrecision(llm=ragas_llm, embeddings=ragas_embeddings)
    except TypeError:
        context_precision_metric = ContextPrecision(llm=ragas_llm)

    metrics_to_use = [faithfulness_metric, answer_relevancy_metric, context_precision_metric]

    result = evaluate(
        dataset=dataset,
        metrics=metrics_to_use
    )

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("=" * 60)

    faith_key = 'faithfulness'
    relevancy_key = 'answer_relevancy'
    precision_key = 'context_precision'

    faithfulness_values = [v for v in result[faith_key] if not math.isnan(v)] if result[faith_key] else []
    relevancy_values = [v for v in result[relevancy_key] if not math.isnan(v)] if result[relevancy_key] else []
    precision_values = [v for v in result[precision_key] if not math.isnan(v)] if result[precision_key] else []

    avg_faithfulness = sum(faithfulness_values) / len(faithfulness_values) if faithfulness_values else 0
    avg_relevancy = sum(relevancy_values) / len(relevancy_values) if relevancy_values else float('nan')
    avg_precision = sum(precision_values) / len(precision_values) if precision_values else 0

    print(f"\nFaithfulness (верность ответа): {avg_faithfulness:.4f}")
    if not math.isnan(avg_relevancy):
        print(f"Answer Relevancy (релевантность ответа): {avg_relevancy:.4f}")
    else:
        print(f"Answer Relevancy (релевантность ответа): не удалось вычислить")
    print(f"Context Precision (точность контекста): {avg_precision:.4f}")

    print("\n" + "=" * 60)
    print("ДЕТАЛИ ПО ВОПРОСАМ")
    print("=" * 60)

    for i, question in enumerate(EVALUATION_QUESTIONS):
        print(f"\nВопрос {i+1}: {question}")
        print(f"  Faithfulness: {result[faith_key][i]:.4f} ---точность ответа")
        rel_val = result[relevancy_key][i]
        if not math.isnan(rel_val):
            print(f"  Answer Relevancy: {rel_val:.4f} ---релевантность ответа вопросу")
        else:
            print(f"  Answer Relevancy: не удалось вычислить ---релевантность ответа вопросу")
        print(f"  Context Precision: {result[precision_key][i]:.4f} ---точность выбранного контекста")

    print("\n" + "=" * 60)
    print("Оценка завершена!")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_rag_system()
