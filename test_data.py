from llm_formater import format_runs_for_llm
from mlops_tool_functions import get_runs_summary
from mlflow import MlflowClient


client = MlflowClient()
experiment_name = "iris_demo"
experiment_id = client.get_experiment_by_name(experiment_name)._experiment_id
top_n = 6
runs = get_runs_summary(experiment_id, top_n=top_n)
runs_text = format_runs_for_llm(runs, max_runs=5)
print('--------------------')
print('runs:--', runs)
print('--------------------')
print('runs_text:--', runs_text, type(runs_text))
