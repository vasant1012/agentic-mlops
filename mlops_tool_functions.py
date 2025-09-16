from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from mlflow import MlflowClient
# used only as a type/sample; we will supply our custom LLM wrapper


@dataclass
class RunSummary:
    run_id: str
    status: str
    params: Dict[str, Any]
    metrics: Dict[str, float]
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    tags: Dict[str, Any] = None


def get_runs_summary(experiment_id: str, top_n: int = 20) -> List[Dict[str, Any]]:  # NOQA E501
    """
    Query MLflow for runs in an experiment and return structured summaries.
    """
    client = MlflowClient()
    # search_runs expects a list of experiment_ids
    # sorts by start_time descending (most recent first)
    runs = client.search_runs([experiment_id], order_by=[
                              "attributes.start_time DESC"], max_results=top_n)
    out = []
    for r in runs:
        params = dict(r.data.params)
        metrics = dict(r.data.metrics)
        status = r.info.status
        start = r.info.start_time
        end = r.info.end_time
        tags = dict(r.data.tags) if r.data.tags else {}
        out.append({
            "run_id": r.info.run_id,
            "status": status,
            "params": params,
            "metrics": metrics,
            "start_time": start,
            "end_time": end,
            "tags": tags,
        })
    return out
