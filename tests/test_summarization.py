from textlite import summarize_text
import pytest


def test_summarize_text_prefers_central_sentences(fake_embedder, token_counter):
    text = (
        "Sentence embeddings help compare meaning across sentences. "
        "They are useful for semantic search and summarization. "
        "The weather is sunny today."
    )

    summary = summarize_text(
        text,
        max_sentences=2,
        embedder=fake_embedder,
        token_counter=token_counter,
    )

    assert "Sentence embeddings help compare meaning across sentences." in summary
    assert "They are useful for semantic search and summarization." in summary
    assert "The weather is sunny today." not in summary


@pytest.mark.parametrize("text", [
    "word " * 2500,
    "word " * 2500 + ". Another sentence.",
    "Repeated sentence. " * 200,
])
def test_summarize_text_enforces_budget(text, fake_embedder, token_counter):
    summary = summarize_text(
        text, max_tokens=20, max_sentences=100,
        embedder=fake_embedder, token_counter=token_counter,
    )
    assert summary
    assert token_counter.count_tokens(summary) <= 20


def test_summarize_text_counts_joined_output(fake_embedder):
    class CharacterCounter:
        def count_tokens(self, text):
            return len(text)

    summary = summarize_text(
        "First sentence. Second sentence. Third sentence.",
        max_sentences=3, max_tokens=30,
        embedder=fake_embedder, token_counter=CharacterCounter(),
    )
    assert len(summary) <= 30


def test_summarize_text_rejects_invalid_budget():
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        summarize_text("A sentence.", max_tokens=0)
