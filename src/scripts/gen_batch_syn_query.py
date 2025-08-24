from mteb import NQ

from src.batch_syn_query import BatchSynQueryGenerator
from src.prompts import BASE_SYS_PROMPT


if __name__ == '__main__':
    nq = NQ()
    nq.load_data()

    for split in nq.eval_splits:
        batch_syn_gen = BatchSynQueryGenerator(
            model='llama-3.1-8b-instant',
            sys_prompt=BASE_SYS_PROMPT,
            batch_path=f'./{split}-batch-syn-queries.jsonl'
        )

        corpus = nq.corpus[split]

        batch_syn_gen.gen(corpus)
