import os
import json
from utils.logger import logger
from utils.mlflow_utils import get_runs_summary, format_runs_for_llm


def main():
    json_file = "mlflow_experiments.json"
    if os.path.exists(json_file):
        logger.info("MLFlow json logs are available!")
    else:
        logger.info(
            "MLFlow json logs are not available. So creating logs in json format."  # NOQA E501
        )
        experiment_name = "iris_demo"
        runs = get_runs_summary(experiment_name, 5)
        exp_dict = format_runs_for_llm(runs, 5)
        with open(json_file, "w") as json_file:
            json.dump(exp_dict, json_file, indent=4)
    print("Choose implementation:")
    print("1. LangChain Agent")
    print("2. LLM Analyzer")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        from agents.langchain_agent import MLflowAnalysisAgent

        agent = MLflowAnalysisAgent(
            model_name="Qwen/Qwen2-1.5B-Instruct")
        result = agent.analyze_from_json(json_file)
        print(result)
    elif choice == "2":
        from agents.direct_analyzer import DirectTransformerMLflowAnalyzer

        analyzer = DirectTransformerMLflowAnalyzer(
            model_name="Qwen/Qwen2-1.5B-Instruct"
        )
        result = analyzer.analyze_json_file(json_file)
        with open('analyzer_result.json', "w") as json_file:
            json.dump(result, json_file, indent=4)
        print(result)
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
