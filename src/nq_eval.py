from mteb import MTEB, NQ
from sentence_transformers import SentenceTransformer
from rich import print

if __name__ == '__main__':
    model = SentenceTransformer('all-MiniLM-L6-v2')
    tasks = [NQ()]
    evaluator = MTEB(tasks)
    res = evaluator.run(model, encode_kwargs={'batch_size': 256})
    print(res)
