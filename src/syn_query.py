import json
import asyncio
import os.path
import logging
from pathlib import Path

from openai import AsyncOpenAI, OpenAI
from openai.types.responses import Response


class SyntheticQueryGenerator:
    """
    Generates synthetic query/ies for the provided ``MTEB`` document corpus and caches them in a ``jsonl`` format file.
    """
    sys_prompt = '''
    You are a helpful AI assistant.
    Generate most relevant and short search question for the document excerpt provided by the user.
    Return only the generated query.
    '''.strip()

    def __init__(
            self,
            client: AsyncOpenAI | OpenAI,
            model: str,
            cache_path: Path = './syn-queries.jsonl'
    ):
        self.client = client
        self.model = model

        if not os.path.exists(cache_path):
            raise ValueError(f"Irrelevant path provided: {cache_path}")
        self.cache_path = cache_path

    @staticmethod
    def _flatten_openai_resp(response: Response):
        return response.output[1].content[0].text.lower()

    def _gen_query(self, doc: str, **kwargs) -> str:
        raise NotImplementedError

    async def _agen_query(self, doc: str, **kwargs) -> str:
        res = await self.client.responses.create(
            model=self.model,
            input=[
                {
                    'role': 'system',
                    'content': self.sys_prompt
                },
                {
                    'role': 'user',
                    'content': doc
                }
            ],
            **kwargs
        )

        return self._flatten_openai_resp(res)

    def _store_jsonl(self, syn_queries: dict[str, str]):
        json_queries = [json.dumps(q) for q in syn_queries]
        with open(self.cache_path) as file:
            file.writelines(json_queries)

    def gen(
            self,
            corpus: dict[str, str]
    ):
        raise NotImplementedError

    async def agen(
            self,
            corpus: dict[str, str],
            batch_size: int = 512,
            **kwargs
    ) -> dict[str, str]:
        if not isinstance(self.client, AsyncOpenAI):
            raise ValueError("Asynchronous method requires an async OpenAI client, got sync client instead.")
        syn_queries = {}

        # sort corpus to hit rate limits early for the provided batch size
        corpus = list(sorted(corpus.items(), key=lambda x: len(x[1]), reverse=True))  # [(doc_id, content), ...]

        itr = range(0, len(corpus), batch_size)
        for i in range(itr):
            batch = corpus[i:i + batch_size]
            logging.info(f'Processing batch: {i}-{i + batch_size}')

            tasks = [self._agen_query(doc[1], **kwargs) for doc in batch]
            results = asyncio.gather(*tasks)

            # map results to relevant doc_id's
            for doc, res in zip(batch, results):
                syn_queries[doc[0]] = res  # {doc_id: syn_query, ...}

        # store the results in jsonl file
        self._store_jsonl(syn_queries)

        return syn_queries
