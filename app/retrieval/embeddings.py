from __future__ import annotations

from typing import List

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

VECTOR_SIZE = 384
_vectorizer = HashingVectorizer(
    n_features=VECTOR_SIZE,
    alternate_sign=False,
    norm="l2",
    ngram_range=(1, 2),
)


def embed_texts(texts: List[str], model_name: str | None = None) -> np.ndarray:
    if not texts:
        return np.empty((0, VECTOR_SIZE), dtype=np.float32)
    embeddings = _vectorizer.transform(texts)
    return embeddings.astype(np.float32).toarray()
