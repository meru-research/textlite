from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Protocol, Sequence

import numpy as np


class PassageScorer(Protocol):
	"""Protocol for query-passage relevance scorers."""

	def score(self, query: str, passages: Sequence[str]) -> Sequence[float]:
		"""Return one relevance score per passage."""


@dataclass(slots=True)
class CrossEncoderReranker:
	"""Rerank passages with a sentence-transformers CrossEncoder model."""

	model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
	_model: Any | None = None

	def _load_model(self) -> Any:
		if self._model is None:
			try:
				from sentence_transformers import CrossEncoder
			except ImportError as exc:
				raise ImportError(
					"sentence-transformers is required. Install dependencies from "
					"textlite/requirements.txt"
				) from exc
			self._model = CrossEncoder(self.model_name)
		return self._model

	def score(self, query: str, passages: Sequence[str]) -> List[float]:
		"""Score each passage for its relevance to the query."""

		if not passages:
			return []

		pairs = [(query, passage) for passage in passages]
		predictions = self._load_model().predict(
			pairs,
			convert_to_numpy=True,
			show_progress_bar=False,
		)
		scores = np.asarray(predictions, dtype=float).reshape(-1)
		if len(scores) != len(passages):
			raise ValueError("CrossEncoder must return one score per passage")
		return scores.tolist()

	def rerank(
		self,
		query: str,
		passages: Sequence[str],
		*,
		top_k: int | None = None,
	) -> List[str]:
		"""Return passages ordered from most to least relevant."""

		return rerank_passages(query, passages, top_k=top_k, scorer=self)


def rerank_passages(
	query: str,
	passages: Sequence[str],
	*,
	top_k: int | None = None,
	scorer: PassageScorer | None = None,
) -> List[str]:
	"""Rerank passages for a query using cross-encoder relevance scores.

	Equal-scoring passages retain their original relative order.
	"""

	if top_k is not None and top_k < 0:
		raise ValueError("top_k must be non-negative")
	if not passages or top_k == 0:
		return []

	scorer = scorer or CrossEncoderReranker()
	scores = list(scorer.score(query, passages))
	if len(scores) != len(passages):
		raise ValueError("scorer must return one score per passage")

	ranked_indices = sorted(range(len(passages)), key=lambda idx: (-scores[idx], idx))
	if top_k is not None:
		ranked_indices = ranked_indices[:top_k]
	return [passages[idx] for idx in ranked_indices]
