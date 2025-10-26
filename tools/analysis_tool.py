from langchain.tools import tool
import json
from typing import Dict, Any

@tool
def analyze_experiment_performance(runs_data: str) -> str:
    """
    Generate an analysis prompt for MLflow experiment runs to guide LLM-based evaluation.

    This function takes formatted MLflow run data (typically produced by
    `load_mlflow_experiments_from_json` or `query_mlflow_experiment`) and constructs
    a structured analysis request. The returned string can be passed to a Large
    Language Model (LLM) to perform higher-level reasoning and generate insights.

    The analysis request instructs the LLM to cover the following aspects:
      1. Performance summary
      2. Best performing runs
      3. Parameter trends
      4. Actionable recommendations
      5. Notable anomalies or patterns

    Args:
        runs_data (str): JSON-formatted string of MLflow experiment run summaries.

    Returns:
        str: A structured analysis prompt string ready for LLM input.

    Example:
        >>> runs = '{"run_0": {"run_id": "abc", "params": {"lr": 0.01}, "metrics": {"accuracy": 0.92}}}'
        >>> analyze_experiment_performance(runs)
        'Please analyze the following MLflow experiment runs and provide insights: ...'
    """
    analysis_prompt = f"""
    Please analyze the following MLflow experiment runs and provide insights:

    {runs_data}

    Include:
    1. Performance summary
    2. Best performing runs
    3. Parameter trends
    4. Recommendations
    5. Anomalies or patterns
    """
    return analysis_prompt


@tool("read_mlflow_logs", return_direct=True)
def read_mlflow_logs(file_path: str) -> Dict[str, Any]:
    """
    Reads MLflow logs from a dumped JSON file and returns structured run data.

    Args:
        file_path (str): Path to the dumped MLflow JSON log file.

    Returns:
        Dict[str, Any]: Parsed MLflow log data with useful info such as metrics, params, and tags.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            log_data = json.load(f)

        # Extract common MLflow sections (if they exist)
        run_info = log_data.get("run_info", {})
        params = log_data.get("params", {})
        metrics = log_data.get("metrics", {})
        tags = log_data.get("tags", {})

        summary = {
            "run_id": run_info.get("run_id", "N/A"),
            "experiment_id": run_info.get("experiment_id", "N/A"),
            "status": run_info.get("status", "N/A"),
            "start_time": run_info.get("start_time", "N/A"),
            "end_time": run_info.get("end_time", "N/A"),
            "params": params,
            "metrics": metrics,
            "tags": tags,
        }

        return {
            "message": f"✅ Successfully read MLflow log from {file_path}",
            "summary": summary
        }

    except FileNotFoundError:
        return {"error": f"❌ File not found: {file_path}"}
    except json.JSONDecodeError:
        return {"error": f"❌ Invalid JSON file: {file_path}"}
    except Exception as e:
        return {"error": f"❌ Unexpected error: {str(e)}"}

