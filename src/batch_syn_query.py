import json
from pathlib import Path

from src.prompts import BASE_SYS_PROMPT


class BatchSynQueryGenerator:
    """
    Generates a JSONL file for asynchronous batch processing (an option available from many LLM API providers),
    which typically costs about half as much but may take several days to complete.

    Given the typical size of evaluation datasets, this approach can help save a significant amount of money.
    """

    def __init__(
            self,
            model: str,
            sys_prompt: str = BASE_SYS_PROMPT,
            batch_path: Path = Path('./batch-syn-queries.jsonl')
    ):
        self.model = model
        self.sys_prompt = sys_prompt
        self.batch_path = batch_path

    def _store_jsonl(self, calls: list[dict]):
        with open(self.batch_path, 'w') as file:
            for i, req in enumerate(calls):
                if i != len(calls) - 1:
                    file.write(json.dumps(req) + '\n')
                else:
                    file.write(json.dumps(req))

    def gen(
            self,
            corpus: dict[str, str]
    ) -> list[dict]:
        """
        Forms a JSONL file for batch completions. Definitely works for [groq](https://groq.com).
        """
        calls = []
        for key, val in corpus.items():  # corpus = {'doc_id': 'content', ...}
            req = {
                'custom_id': key,
                'method': 'POST',
                'url': '/v1/chat/completions',
                'body': {
                    'model': self.model,
                    'messages': [
                        {
                            'role': 'system',
                            'content': self.sys_prompt
                        },
                        {
                            'role': 'user',
                            'content': val
                        }
                    ]
                }
            }

            calls.append(req)

        self._store_jsonl(calls)

        return calls
