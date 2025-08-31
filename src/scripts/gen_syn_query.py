import asyncio
import logging

from mteb import NQ
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

from src.syn_query import SyntheticQueryGenerator
from src.prompts import BASE_SYS_PROMPT

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    load_dotenv()
    client = AsyncOpenAI(
        base_url='https://api.groq.com/openai/v1'
    )

    nq = NQ()
    nq.load_data()

    for split in nq.eval_splits:
        syn_gen = SyntheticQueryGenerator(
            client=client,
            model='llama-3.1-8b-instant',
            cache_path=f'./{split}-syn-queries.jsonl',
            sys_prompt=BASE_SYS_PROMPT
        )

        corpus = dict(list(nq.corpus[split].items())[:16])

        asyncio.run(
            syn_gen.agen(
                corpus,
                batch_size=8,
                temperature=0.7
            )
        )
