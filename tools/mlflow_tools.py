import json
from langchain.tools import tool
from utils.mlflow_utils import get_runs_summary, format_runs_for_llm


@tool
def load_mlflow_experiments_from_json(file_path: str) -> str:
    """
    Load MLflow experiment runs from a JSON file and format them for LLM analysis.

    This function reads a JSON file containing experiment run data in one of the following formats:
      - A list of run dictionaries.
      - A dictionary with a "runs" or "experiments" key holding a list of runs.
      - A single run dictionary.

    Each run is normalized to include only the run ID, parameters, and metrics. The
    resulting list of runs is then passed to `format_runs_for_llm`, which formats
    and writes the processed runs into a new `mlflow_experiments.json` file.

    Args:
        file_path (str): Path to the JSON file containing MLflow experiment run data.

    Returns:
        str: JSON string of the formatted experiment runs if successful,
             otherwise an error message (e.g., invalid JSON structure or file errors).

    Example:
        >>> load_mlflow_experiments_from_json("mlflow_experiments.json")
        '{"run_0": {"run_id": "...", "params": {...}, "metrics": {...}}, ...}'
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            runs = data
        elif isinstance(data, dict):
            runs = data.get("runs", data.get("experiments", [data]))
        else:
            return "Error: Invalid JSON structure."

        formatted_runs = []
        for run in runs:
            formatted_runs.append(
                {
                    "run_id": run.get("run_id", "unknown"),
                    "params": run.get("params", {}),
                    "metrics": run.get("metrics", {}),
                }
            )
        return format_runs_for_llm(formatted_runs)

    except Exception as e:
        return f"Error loading JSON: {str(e)}"


@tool
def query_mlflow_experiment(experiment_name: str, top_n: int = 20) -> str:
    """
    Query an MLflow experiment by name and return a formatted summary of its runs.

    This function connects to the active MLflow tracking server and retrieves
    the most recent runs for the specified experiment. The runs are enriched with
    metadata such as run ID, status, parameters, metrics, start/end time, and tags.
    Results are then formatted for LLM-friendly analysis using `format_runs_for_llm`.

    Args:
        experiment_name (str): The name of the MLflow experiment to query.
        top_n (int, optional): Maximum number of runs to retrieve (default is 20).

    Returns:
        str: JSON string of formatted experiment runs if successful,
             otherwise an error message (e.g., experiment not found or MLflow errors).

    Example:
        >>> query_mlflow_experiment("my_experiment", top_n=5)
        '{"run_0": {"run_id": "...", "params": {...}, "metrics": {...}}, ...}'
    """
    try:
        runs = get_runs_summary(experiment_name, top_n)
        return format_runs_for_llm(runs)
    except Exception as e:
        return f"Error querying MLflow: {str(e)}"
