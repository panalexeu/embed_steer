from typing import Any

from mteb import HFSubset, ScoresDict
from mteb.abstasks import AbsTaskRetrieval, RetrievalEvaluator
from mteb.evaluation.evaluators import DenseRetrievalExactSearch


class EmbedSteerDenseRetrievalExactSearch(DenseRetrievalExactSearch):
    def search(
            self,
            corpus: dict[str, dict[str, str]],
            queries: dict[str, str | list[str]],
            top_k: int, task_name: str,
            instructions: dict[str, str] | None = None,
            request_qid: str | None = None, return_sorted: bool = False,
            **kwargs
    ) -> dict[str, dict[str, float]]:
        return super().search(corpus, queries, top_k, task_name, instructions, request_qid, return_sorted, **kwargs)


class EmbedSteerRetrievalEvaluator(RetrievalEvaluator):

    def __init__(
            self,
            retriever,
            task_name: str | None = None,
            k_values: list[int] = [1, 3, 5, 10, 20, 100, 1000],
            encode_kwargs: dict[str, Any] = {}, **kwargs
    ):
        super().__init__(retriever, task_name, k_values, encode_kwargs, **kwargs)
        self.retriever = EmbedSteerDenseRetrievalExactSearch(
            retriever, encode_kwargs=encode_kwargs, **kwargs
        )


class EmbedSteerTaskRetrieval(AbsTaskRetrieval):
    """AbsTaskRetrieval is a base class of NQ and MSMARCO benchmarks in MTEB"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate(
            self,
            model,
            split: str = "test",
            subsets_to_run: list[HFSubset] | None = None,
            *,
            encode_kwargs: dict[str, Any] = {},
            **kwargs
    ) -> dict[HFSubset, ScoresDict]:
        retriever = EmbedSteerRetrievalEvaluator(
            retriever=model,
            task_name=self.metadata.name,
            encode_kwargs=encode_kwargs,
            **kwargs
        )
        return super().evaluate(model, split, subsets_to_run, encode_kwargs=encode_kwargs, **kwargs)
