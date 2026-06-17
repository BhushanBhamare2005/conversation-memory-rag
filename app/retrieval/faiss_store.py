from __future__ import annotations

import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from app.models import RetrievedDocument
from app.retrieval.embeddings import embed_texts


class FaissMemoryStore:
    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self.vectors = np.empty((0, dimension), dtype=np.float32)
        self.documents: List[RetrievedDocument] = []

    def add_documents(self, documents: Sequence[Dict[str, Any]], model_name: str | None = None) -> None:
        if not documents:
            return
        texts = [document["content"] for document in documents]
        vectors = embed_texts(texts, model_name=model_name)
        if vectors.size == 0:
            return
        if self.vectors.size == 0:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        for document in documents:
            self.documents.append(
                RetrievedDocument(
                    doc_id=document["doc_id"],
                    doc_type=document["doc_type"],
                    content=document["content"],
                    score=0.0,
                    metadata=document.get("metadata", {}),
                )
            )

    def search(self, query: str, top_k: int = 5, model_name: str | None = None) -> List[RetrievedDocument]:
        if not self.documents or self.vectors.size == 0:
            return []
        query_vector = embed_texts([query], model_name=model_name)
        if query_vector.size == 0:
            return []
        scores = query_vector @ self.vectors.T
        ranked_indices = np.argsort(scores[0])[::-1][: min(top_k, len(self.documents))]
        results: List[RetrievedDocument] = []
        for index in ranked_indices:
            if index < 0 or index >= len(self.documents):
                continue
            document = self.documents[index]
            results.append(
                RetrievedDocument(
                    doc_id=document.doc_id,
                    doc_type=document.doc_type,
                    content=document.content,
                    score=float(scores[0][index]),
                    metadata=document.metadata,
                )
            )
        return results

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors)
        with (path / "documents.pkl").open("wb") as handle:
            pickle.dump([asdict(document) for document in self.documents], handle)

    @classmethod
    def load(cls, directory: str | Path) -> "FaissMemoryStore":
        path = Path(directory)
        store = cls()
        vectors_path = path / "vectors.npy"
        if vectors_path.exists():
            store.vectors = np.load(vectors_path)
        with (path / "documents.pkl").open("rb") as handle:
            raw_documents = pickle.load(handle)
        store.documents = [RetrievedDocument(**document) for document in raw_documents]
        return store
