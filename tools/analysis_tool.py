from langchain.tools import tool

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
