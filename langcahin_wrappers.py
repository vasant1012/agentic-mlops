import json
# LangChain imports (tools + agent)
from langchain.agents import Tool
from local_hugging_face import LocalHFAdapter
from llm_formater import format_runs_for_llm
from mlops_tool_functions import get_runs_summary
from prompt import SUMMARY_PROMPT, RECOMMEND_PROMPT
from logger import logger


def make_tools(llm_adapter: LocalHFAdapter):
    def summarize_runs_tool(experiment_id: str, top_n: int = 2) -> str:
        runs = get_runs_summary(experiment_id, top_n=top_n)
        runs_text = format_runs_for_llm(runs, max_runs=5)
        prompt = SUMMARY_PROMPT.format(runs_text=runs_text)
        logger.info("DEBUG: Sending prompt to LLM (summary) — length:", len(prompt))  # NOQA E501
        return llm_adapter(prompt)

    def recommend_hparams_tool(experiment_id: str, top_n: int = 5) -> str:
        runs = get_runs_summary(experiment_id, top_n=top_n)
        runs_text = format_runs_for_llm(runs, max_runs=5)
        prompt = RECOMMEND_PROMPT.format(runs_text=runs_text)
        logger.info("DEBUG: Sending prompt to LLM (recommend) — length:", len(prompt))  # NOQA E501
        return llm_adapter(prompt)

    # Wrap in LangChain Tool objects
    tools = [
        Tool(
            name="summarize_runs",
            func=summarize_runs_tool,
            description="Summarize recent MLflow runs of an experiment. Args: experiment_id, top_n (optional)."  # NOQA E501
        ),
        Tool(
            name="recommend_hparams",
            func=recommend_hparams_tool,
            description="Recommend next hyperparameter trials. Args: experiment_id, top_n (optional)."  # NOQA E501
        ),
        Tool(
            name="get_runs_summary",
            func=lambda experiment_id, top_n=20: json.dumps(
                get_runs_summary(experiment_id, top_n=top_n), default=str),
            description="Return raw MLflow runs JSON for programmatic use. Args: experiment_id, top_n (optional)."  # NOQA E501
        )
    ]
    return tools
