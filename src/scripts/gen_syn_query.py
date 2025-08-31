import asyncio
import logging
import sys

from mteb import NQ
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

from src.syn_query import SyntheticQueryGenerator
from src.prompts import BASE_SYS_PROMPT
from src.utils import _truncate_corpus

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    url = sys.argv[1]
    model = sys.argv[2]
    tokens = sys.argv[3]

    load_dotenv()
    client = AsyncOpenAI(
        base_url=url,
    )

    nq = NQ()
    nq.load_data()

    for split in nq.eval_splits:
        syn_gen = SyntheticQueryGenerator(
            client=client,
            model=model,
            cache_path=f'./{split}-syn-queries.jsonl',
            sys_prompt=BASE_SYS_PROMPT
        )

        corpus = dict(list(nq.corpus[split].items())[:16])
        corpus = _truncate_corpus(model, corpus, token_limit=tokens)

        asyncio.run(
            syn_gen.agen(
                corpus,
                batch_size=8,
                temperature=0.7
            )
        )
