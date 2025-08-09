import mteb
from mteb.tasks import NQ, MSMARCO
from sentence_transformers import SentenceTransformer

# DEBUG
# from mteb.overview import create_task_list
# tasks = create_task_list()
# breakpoint()


# ====
task = mteb.get_tasks(tasks=['NQ'])
evaluator = mteb.MTEB(task)
model = SentenceTransformer(model_name_or_path='all-MiniLM-L6-v2')
evaluator.run(model)




