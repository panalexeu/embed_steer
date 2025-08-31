from transformers import AutoTokenizer


def _truncate_corpus(
        model_name: str,
        corpus: dict[str, str],
        token_limit: int = 512
) -> dict[str, str]:
    """Truncates corpus documents so they don’t exceed the token limit."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for key, text in corpus.items():
        tokens = tokenizer.encode(text, truncation=True, max_length=token_limit)
        corpus[key] = tokenizer.decode(tokens, skip_special_tokens=True)

    return corpus
