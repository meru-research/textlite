import pytest

from textlite.reranking import CrossEncoderReranker, rerank_passages


class FakeScorer:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def score(self, query, passages):
        self.calls.append((query, list(passages)))
        return self.scores


def test_rerank_passages_orders_by_relevance_and_preserves_ties():
    scorer = FakeScorer([0.2, 0.9, 0.9])
    passages = ["first", "second", "third"]

    ranked = rerank_passages("query", passages, scorer=scorer)

    assert ranked == ["second", "third", "first"]
    assert scorer.calls == [("query", passages)]


def test_rerank_passages_applies_top_k_without_scoring_empty_input():
    scorer = FakeScorer([0.1, 0.8, 0.4])

    assert rerank_passages("query", ["a", "b", "c"], top_k=2, scorer=scorer) == ["b", "c"]
    assert rerank_passages("query", [], scorer=scorer) == []
    assert len(scorer.calls) == 1


def test_rerank_passages_rejects_invalid_results():
    with pytest.raises(ValueError, match="top_k"):
        rerank_passages("query", ["passage"], top_k=-1)

    with pytest.raises(ValueError, match="one score per passage"):
        rerank_passages("query", ["a", "b"], scorer=FakeScorer([0.5]))


def test_cross_encoder_builds_query_passage_pairs():
    class FakeCrossEncoder:
        def __init__(self):
            self.pairs = None

        def predict(self, pairs, **kwargs):
            self.pairs = pairs
            assert kwargs == {"convert_to_numpy": True, "show_progress_bar": False}
            return [0.25, 0.75]

    model = FakeCrossEncoder()
    reranker = CrossEncoderReranker(_model=model)

    assert reranker.rerank("capital of France", ["Berlin", "Paris"]) == ["Paris", "Berlin"]
    assert model.pairs == [
        ("capital of France", "Berlin"),
        ("capital of France", "Paris"),
    ]