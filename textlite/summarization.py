from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .deduplication import deduplicate_texts
from .embeddings import SentenceTransformerEmbedder, TokenCounter
from .types import Embedder
from .utils import cosine_similarity_matrix, cosine_similarity_to_vector, split_sentences


def _mmr_order(similarity_to_centroid: np.ndarray, pairwise_similarity: np.ndarray, diversity: float) -> List[int]:
    remaining = set(range(len(similarity_to_centroid)))
    selected: List[int] = []

    while remaining:
        best_idx = None
        best_score = None
        for idx in remaining:
            if not selected:
                score = similarity_to_centroid[idx]
            else:
                redundancy = max(pairwise_similarity[idx, chosen] for chosen in selected)
                score = (1.0 - diversity) * similarity_to_centroid[idx] - diversity * redundancy
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx
        assert best_idx is not None
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected



def summarize_text(
    text: str,
    *,
    max_sentences: int = 3,
    max_tokens: int | None = None,
    diversity: float = 0.1,
    embedder: Embedder | None = None,
    token_counter: TokenCounter | None = None,
) -> str:
    """Sentence-BERT style extractive summarization."""

    sentences = split_sentences(text)
    if not sentences:
        return ""

    deduped = deduplicate_texts(sentences, threshold=0.98, embedder=embedder)
    candidates = deduped.unique_texts
    if len(candidates) <= 1:
        return candidates[0] if candidates else ""

    embedder = embedder or SentenceTransformerEmbedder()
    embeddings = embedder.encode(candidates)
    centroid = embeddings.mean(axis=0)
    centroid_similarity = cosine_similarity_to_vector(embeddings, centroid)
    pairwise_similarity = cosine_similarity_matrix(embeddings)
    order = _mmr_order(centroid_similarity, pairwise_similarity, diversity=diversity)

    token_counter = token_counter or TokenCounter()
    chosen_indices: List[int] = []
    total_tokens = 0
    for idx in order:
        sentence = candidates[idx]
        sentence_tokens = token_counter.count_tokens(sentence)
        if max_tokens is not None and chosen_indices and total_tokens + sentence_tokens > max_tokens:
            continue
        chosen_indices.append(idx)
        total_tokens += sentence_tokens
        if max_tokens is None and len(chosen_indices) >= max_sentences:
            break

    if not chosen_indices:
        chosen_indices = [order[0]]

    chosen_sentences = [candidates[idx] for idx in sorted(chosen_indices)]
    return " ".join(chosen_sentences)



def summarize_paragraphs(
    paragraphs: Sequence[str],
    *,
    max_sentences_per_paragraph: int = 2,
    max_tokens: int | None = None,
    embedder: Embedder | None = None,
    token_counter: TokenCounter | None = None,
) -> List[str]:
    summaries: List[str] = []
    running_tokens = 0
    token_counter = token_counter or TokenCounter()
    for paragraph in paragraphs:
        summary = summarize_text(
            paragraph,
            max_sentences=max_sentences_per_paragraph,
            max_tokens=max_tokens,
            embedder=embedder,
            token_counter=token_counter,
        )
        if not summary:
            continue
        summary_tokens = token_counter.count_tokens(summary)
        if max_tokens is not None and summaries and running_tokens + summary_tokens > max_tokens:
            break
        summaries.append(summary)
        running_tokens += summary_tokens
    return summaries
