# RAG с ChromaDB и RAGAS

## Overview
Учебный проект RAG-системы (Retrieval-Augmented Generation) с использованием ChromaDB для векторного хранилища и RAGAS для оценки качества.

## Project Architecture
```
5-6-rag_qality/
├── data/              # Исходные документы (doc1.txt, doc2.txt)
├── chroma_db/         # Локальная база ChromaDB
├── ingest.py          # Скрипт индексации документов
├── rag_assistant.py   # RAG-ассистент (интерактивный CLI)
├── evaluate_rag.py    # Оценка через RAGAS
├── config.py          # Конфигурация
├── requirements.txt   # Зависимости
└── .env               # Не используется (ключи хранятся в Replit Secrets)
```

## Key Configuration
- OpenAI API key stored in Replit Secrets (accessed via `os.getenv("OPENAI_API_KEY")`)
- Embedding model: text-embedding-3-small
- Chat model: gpt-3.5-turbo
- Chunk size: 500 chars, overlap: 100 chars, top_k: 5

## Usage
1. `python ingest.py` — индексация документов из data/
2. `python rag_assistant.py` — интерактивный ассистент
3. `python evaluate_rag.py` — оценка качества через RAGAS

## Recent Changes
- 2026-02-13: Set up Replit Secrets for OPENAI_API_KEY, installed dependencies, indexed documents, configured workflow
