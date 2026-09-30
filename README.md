# textlite

`textlite` is a lightweight package for semantic clustering, deduplication,
extractive summarization, and context compression using BERT/Sentence Transformer
style models.

## Capabilities

- Entity clustering based on semantic similarity
- Sentence-BERT style extractive summarization
- Deduplication of words, phrases, sentences, or paragraphs
- Context compression under a token budget
- Query-aware sentence extraction under a token budget
- Query-passage reranking with a cross-encoder model

## Install

```bash
pip install -r textlite/requirements.txt
pip install -e .
```

## Example

```python
from textlite import (
    cluster_entities,
    compress_contexts,
    compress_query_contexts,
    deduplicate_texts,
    rerank_passages,
    summarize_text,
)

entities = [
    "large language model",
    "LLM",
    "neural search",
    "semantic retrieval",
]
clusters = cluster_entities(entities)

unique = deduplicate_texts([
    "Apple releases earnings tomorrow.",
    "Apple releases earnings tomorrow!",
    "Microsoft expands Azure AI offerings.",
])

summary = summarize_text(
    "Sentence embeddings help compare meaning across sentences. "
    "They are useful for semantic search and summarization."
)

compressed = compress_contexts([
    "Paragraph 1 ...",
    "Paragraph 2 ...",
], num_tokens=80)

query_contexts = compress_query_contexts(
    "How does retrieval augmented generation work?",
    ["Paragraph 1 ...", "Paragraph 2 ..."],
    num_tokens=80,
)

ranked = rerank_passages(
    "What is the capital of France?",
    ["Berlin is Germany's capital.", "Paris is the capital of France."],
)
```
