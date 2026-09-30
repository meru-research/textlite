import numpy as np

from textlite import compress_contexts, compress_query_contexts
from textlite.embeddings import TokenCounter


def test_compress_contexts_stays_within_budget_and_keeps_relevant_content(fake_embedder):
    paragraphs = [
        "Retrieval augmented generation combines search with generation.",
        "RAG systems use retrieval to ground answers with documents.",
        "Vector databases speed up semantic retrieval over embeddings.",
        "Gardening tools are on sale this weekend.",
    ]
    token_counter = TokenCounter()

    compressed = compress_contexts(
        paragraphs,
        num_tokens=20,
        dedup_threshold=0.9,
        embedder=fake_embedder,
        token_counter=token_counter,
    )

    assert compressed
    assert sum(token_counter.count_tokens(item) for item in compressed) <= 20
    joined = " ".join(compressed)
    assert "retrieval" in joined.lower() or "generation" in joined.lower()


class QueryEmbedder:
    def __init__(self):
        self.vectors = {
            "where should i visit in paris?": [1.0, 0.0],
            "paris is the capital of france.": [1.0, 0.0],
            "the eiffel tower welcomes visitors daily.": [0.9, 0.1],
            "penguins live primarily in the southern hemisphere.": [0.0, 1.0],
        }

    def encode(self, texts):
        vectors = []
        for text in texts:
            key = text.strip().lower()
            if key in self.vectors:
                vectors.append(self.vectors[key])
            elif "paris" in key or "eiffel" in key:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return np.asarray(vectors, dtype=float)


def test_compress_query_contexts_extracts_and_groups_relevant_sentences():
    paris = (
        "Paris is the capital of France. "
        "The Eiffel Tower welcomes visitors daily."
    )
    contexts = compress_query_contexts(
        "Where should I visit in Paris?",
        [paris, paris, "Penguins live primarily in the Southern Hemisphere."],
        num_tokens=20,
        min_similarity=0.5,
        embedder=QueryEmbedder(),
        token_counter=TokenCounter(),
    )

    assert contexts == [paris]
    assert sum(TokenCounter().count_tokens(context) for context in contexts) <= 20


def test_compress_query_contexts_respects_budget_and_empty_inputs():
    contexts = compress_query_contexts(
        "Where should I visit in Paris?",
        [
            "Paris is the capital of France. "
            "The Eiffel Tower welcomes visitors daily."
        ],
        num_tokens=8,
        min_similarity=0.5,
        embedder=QueryEmbedder(),
        token_counter=TokenCounter(),
    )

    assert contexts == ["Paris is the capital of France."]
    assert compress_query_contexts("", ["A paragraph."], 10) == []
    assert compress_query_contexts("query", ["A paragraph."], 0) == []
