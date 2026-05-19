import json
import os
import sys
from pathlib import Path

import requests

from retrieval.chunking import chunk_documents
from retrieval.qdrant_store import QdrantVectorStore


DEFAULT_DOCUMENTS_PATH = Path(__file__).resolve().parents[1] / "data" / "raw_documents.jsonl"
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")
DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


class QAPipeline:
    def __init__(
        self,
        documents_path: str | Path = DEFAULT_DOCUMENTS_PATH,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
    ):
        self.documents_path = Path(documents_path)
        self.model = model
        self.base_url = base_url
        self._store = None

    @property
    def store(self) -> QdrantVectorStore:
        if self._store is None:
            self._store = QdrantVectorStore()
        return self._store

    def load_documents(self) -> list[dict]:
        with self.documents_path.open(encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]

    def build_chunks(self, documents: list[dict] | None = None) -> list[dict]:
        if documents is None:
            documents = self.load_documents()

        all_chunks = []

        for doc in documents:
            source_id = (
                doc.get("source_id")
                or doc.get("url")
                or doc.get("title")
                or doc.get("source", "unknown")
            )

            doc_chunks = chunk_documents([doc])

            for chunk in doc_chunks:
                chunk["source_id"] = source_id
                chunk["chunk_id"] = f"{source_id}_{chunk['chunk_index']}"
                all_chunks.append(chunk)

        return all_chunks

    def index_documents(self, documents: list[dict] | None = None) -> list[dict]:
        chunks = self.build_chunks(documents)
        self.store.upsert_chunks(chunks)
        return chunks

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        return self.store.top_k_retrieve(query, k=k)

    def build_context(self, retrieved_chunks: list[dict]) -> str:
        context_blocks = []

        for index, chunk in enumerate(retrieved_chunks, start=1):
            context_blocks.append(
                "\n".join(
                    [
                        f"[{index}] title: {chunk.get('title', '')}",
                        f"[{index}] source: {chunk.get('source', '')}",
                        f"[{index}] url: {chunk.get('url', '')}",
                        f"[{index}] content: {chunk.get('content', '')}",
                    ]
                )
            )

        return "\n\n".join(context_blocks)

    def chat_with_ollama(self, system_prompt: str, user_prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=120,
        )
        response.raise_for_status()

        data = response.json()
        return data["message"]["content"]

    def synthesize_answer(self, query: str, retrieved_chunks: list[dict]) -> str:
        context = self.build_context(retrieved_chunks)
        prompt = f"Question: {query}\n\nRetrieved context:\n{context}"

        return self.chat_with_ollama(
            system_prompt=(
                "Answer the question using only the provided retrieval context. "
                "If the context is insufficient, say so briefly."
            ),
            user_prompt=prompt,
        )

    def ask(self, query: str, k: int = 5) -> dict:
        self.index_documents()
        retrieved_chunks = self.retrieve(query, k=k)
        answer = self.synthesize_answer(query, retrieved_chunks)

        return {
            "query": query,
            "retrieved_chunks": retrieved_chunks,
            "answer": answer,
        }


def generate_answer(query: str, top_k: int = 5) -> dict:
    pipeline = QAPipeline()
    result = pipeline.ask(query, k=top_k)

    return {
        "query": result["query"],
        "answer": result["answer"],
        "sources": result["retrieved_chunks"],
    }
