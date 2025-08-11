from mteb.abstasks import AbsTaskRetrieval, RetrievalEvaluator
from mteb.evaluation.evaluators import DenseRetrievalExactSearch


class EmbedSteerDenseRetrievalExactSearch(DenseRetrievalExactSearch):
    pass


class EmbedSteerRetrievalEvaluator(RetrievalEvaluator):
    pass


class EmbedSteerRetrieval(AbsTaskRetrieval):
    """AbsTaskRetrieval is a base class of NQ and MSMARCO benchmarks in MTEB"""
    pass
