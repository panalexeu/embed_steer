## embed-steer

### Command Usage

To generate synthetic queries, use the following command:

```bash
python3 -m src.scripts.gen_syn_query <URL> <MODEL> <TOKENS> <BATCH_SIZE>
```

- `<URL>`: OpenAI Chat Completions API compatible server.
- `<MODEL>`: Name of the model to be used.
- `<TOKENS>`: Maximum number of tokens to which each document in the corpus will be truncated.
- `<BATCH_SIZE>`

### Example

Here is an example using a model served via `vLLM`:

```bash
python3 -m src.scripts.gen_syn_query http://localhost:8000/v1 Qwen/Qwen2.5-0.5B-Instruct 192 780
```
