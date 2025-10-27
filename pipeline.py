import json
from typing import Dict, Any


def load_mlflow_experiments_from_json(file_path: str) -> Dict[str, Any]:  # NOQA E501
    """
    Loads MLflow experiments from a JSON file and returns their structured data. # NOQA E501

    Args:
    file_path (str): Path to the dumped MLflow JSON log file.

    Returns:
    Dict[str, Any]: Structured data containing MLflow experiment details.
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def analyze_experiment_performance(runs_data: str) -> str:
    """
    Generates an analysis prompt for MLflow experiment runs based on the given data. # NOQA E501

    Args:
    runs_data (str): JSON-formatted string of MLflow experiment run summaries.

    Returns:
    str: A structured analysis prompt string ready for LLM input.
    """
    # Example parsing logic
    parsed_data = parse_mlflow_run_data(runs_data)
    summary = summarize_runs(parsed_data['runs'])
    best_runs = get_best_runs(parsed_data['runs'], parsed_data['best_runs'])
    key_metrics = calculate_key_metrics(best_runs)
    recommendations = generate_recommendations(key_metrics)

    # Construct the final analysis prompt
    final_prompt = {
        "summary": summary,
        "best_runs": best_runs,
        "key_metrics": key_metrics,
        "recommendations": recommendations
    }

    return json.dumps(final_prompt, indent=4)


def parse_mlflow_run_data(runs_data: str) -> dict:
    """
    Parses the MLflow run data into a dictionary structure.

    Args:
    runs_data (str): JSON-formatted string of MLflow experiment run summaries.

    Returns:
    dict: Dictionary containing MLflow run data.
    """
    parsed_data = {}
    runs = json.loads(runs_data)
    for run_id, run in runs.items():
        parsed_data[run_id] = {
            'run_id': run_id,
            **run.get('params', {}),
            **run.get('metrics', {})
        }
    return parsed_data


def summarize_runs(runs: list) -> dict:
    """
    Summarizes the performance of each run.

    Args:
    runs (list): List of dictionaries representing MLflow runs.

    Returns:
    dict: Summary of runs including metrics and parameters.
    """
    summary = {}
    for run in runs:
        summary[run['run_id']] = {
            'metrics': run['metrics'],
            'params': run['params']
        }
    return summary


def get_best_runs(runs: list, best_runs: list) -> list:
    """
    Retrieves the best-performing runs from the given list of runs.

    Args:
    runs (list): List of dictionaries representing MLflow runs.
    best_runs (list): List of dictionaries representing the best-performing runs. # NOQA E501

    Returns:
    list: Best-performing runs.
    """
    best_runs = sorted(
        runs, key=lambda x: x['metrics']['accuracy'], reverse=True)[:5]
    return best_runs


def calculate_key_metrics(runs: list) -> dict:
    """
    Calculates key metrics for each run.

    Args:
    runs (list): List of dictionaries representing MLflow runs.

    Returns:
    dict: Key metrics for each run.
    """
    key_metrics = {}
    for run in runs:
        key_metrics[run['run_id']] = {
            'accuracy': run['metrics']['accuracy'],
            'time_to_completion': run['metrics'].get('time_to_completion', None), # NOQA E501
            'parameters': run['params']
        }
    return key_metrics


def generate_recommendations(key_metrics: dict) -> str:
    """
    Generates actionable recommendations based on the key metrics.

    Args:
    key_metrics (dict): Dictionary of key metrics for each run.

    Returns:
    str: Recommendations for improving MLflow experiments.
    """
    recommendations = []
    for run_id, metric in key_metrics.items():
        if metric['accuracy'] > 0.8:
            recommendations.append(
                f"Use {metric['parameters']} for better accuracy.")
        elif metric['time_to_completion'] is not None:
            recommendations.append(
                f"Consider optimizing {run_id} for faster completion time.")
    return '\n'.join(recommendations)


# Example usage
if __name__ == "__main__":
    # Sample JSON data
    sample_runs_data = '{"run_0": {"run_id": "abc", "params": {"lr": 0.01}, "metrics": {"accuracy": 0.92}}, "run_1": {"run_id": "def", "params": {"lr": 0.02}, "metrics": {"accuracy": 0.87}}}' # NOQA E501

    # Analyze the sample data
    print(analyze_experiment_performance(sample_runs_data))
