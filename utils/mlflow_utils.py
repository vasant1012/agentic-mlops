import json
from typing import List, Dict, Any
from mlflow import MlflowClient


def format_runs_for_llm(runs: List[Dict[str, Any]], max_runs: int = 10) -> str:
    exp_dict = {}
    for i in range(len(runs[:max_runs])):
        run_dict = {
            "run_id": runs[i]["run_id"],
            "params": runs[i]["params"],
            "metrics": runs[i]["metrics"]
        }
        exp_dict[f"run_{i}"] = run_dict
    with open("mlflow_experiments.json", "w") as json_file:
        json.dump(exp_dict, json_file, indent=4)
    return exp_dict


def get_runs_summary(experiment_name: str, top_n: int = 20) -> List[Dict[str, Any]]: # NOQA E501
    client = MlflowClient()
    experiment_id = client.get_experiment_by_name(
        experiment_name)._experiment_id
    runs = client.search_runs(
        [experiment_id], order_by=["attributes.start_time DESC"], max_results=top_n # NOQA E501
    )
    out = []
    for r in runs:
        out.append({
            "run_id": r.info.run_id,
            "params": dict(r.data.params),
            "metrics": dict(r.data.metrics),
        })
    return out
