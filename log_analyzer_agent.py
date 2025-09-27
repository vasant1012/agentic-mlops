import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from mlflow import MlflowClient


# Data classes and utility functions
@dataclass
class RunSummary:
    run_id: str
    status: str
    params: Dict[str, Any]
    metrics: Dict[str, float]
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    tags: Dict[str, Any] = None


def format_runs_for_llm(runs: List[Dict[str, Any]], max_runs: int = 10) -> str:
    """
    Convert structured runs to a compact text block for prompting.
    """
    exp_dict = {}
    for i in range(len(runs[:max_runs])):
        run_dict = {}
        run_dict["run_id"] = runs[i]["run_id"]
        run_dict["params"] = runs[i]["params"]
        run_dict["metrics"] = runs[i]["metrics"]
        exp_dict[f"run_{i}"] = run_dict
    with open("mlflow_experiments.json", "w") as json_file:
        json.dump(exp_dict, json_file, indent=4)
    return exp_dict


def get_runs_summary(experiment_name: str, top_n: int = 20) -> List[Dict[str, Any]]:
    """
    Query MLflow for runs in an experiment and return structured summaries.
    """
    client = MlflowClient()
    # search_runs expects a list of experiment_ids
    # sorts by start_time descending (most recent first)
    experiment_id = client.get_experiment_by_name(experiment_name)._experiment_id
    runs = client.search_runs(
        [experiment_id], order_by=["attributes.start_time DESC"], max_results=top_n
    )
    out = []
    for r in runs:
        params = dict(r.data.params)
        metrics = dict(r.data.metrics)
        status = r.info.status
        start = r.info.start_time
        end = r.info.end_time
        tags = dict(r.data.tags) if r.data.tags else {}
        out.append(
            {
                "run_id": r.info.run_id,
                "status": status,
                "params": params,
                "metrics": metrics,
                "start_time": start,
                "end_time": end,
                "tags": tags,
            }
        )
    return out


# LangChain Tools
@tool
def load_mlflow_experiments_from_json(file_path: str) -> str:
    """
    Load MLflow experiments data from a JSON file and return formatted content for analysis.

    Args:
        file_path: Path to the JSON file containing MLflow experiments data

    Returns:
        Formatted string containing experiment runs data ready for LLM analysis
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, list):
            runs = data
        elif isinstance(data, dict):
            # Check for common keys that might contain runs data
            if "runs" in data:
                runs = data["runs"]
            elif "experiments" in data:
                runs = data["experiments"]
            else:
                # Assume the dict itself represents a single run
                runs = [data]
        else:
            return "Error: Invalid JSON structure. Expected list or dict."

        # Ensure runs have required fields
        formatted_runs = []
        for run in runs:
            if not isinstance(run, dict):
                continue

            formatted_run = {
                "run_id": run.get("run_id", "unknown"),
                "status": run.get("status", "unknown"),
                "params": run.get("params", {}),
                "metrics": run.get("metrics", {}),
                "start_time": run.get("start_time"),
                "end_time": run.get("end_time"),
                "tags": run.get("tags", {}),
            }
            formatted_runs.append(formatted_run)

        return format_runs_for_llm(formatted_runs)

    except FileNotFoundError:
        return f"Error: File '{file_path}' not found."
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON format - {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def query_mlflow_experiment(experiment_name: str, top_n: int = 20) -> str:
    """
    Query MLflow tracking server for experiment runs and return formatted content for analysis.

    Args:
        experiment_id: MLflow experiment ID
        top_n: Maximum number of runs to retrieve (default: 20)

    Returns:
        Formatted string containing experiment runs data ready for LLM analysis
    """
    try:
        client = MlflowClient()
        experiment_id = client.get_experiment_by_name(experiment_name)._experiment_id
        runs = get_runs_summary(experiment_id, top_n)
        return format_runs_for_llm(runs)
    except Exception as e:
        return f"Error querying MLflow: {str(e)}"


@tool
def analyze_experiment_performance(runs_data: str) -> str:
    """
    Analyze experiment performance and provide insights about the runs.
    This tool expects formatted runs data as input.

    Args:
        runs_data: Formatted string containing runs information

    Returns:
        Analysis prompt for the LLM to process
    """
    analysis_prompt = f"""
    Please analyze the following MLflow experiment runs and provide insights:

    {runs_data}

    Please provide:
    1. Summary of experiment performance
    2. Best performing runs based on metrics
    3. Parameter trends and their impact on performance
    4. Recommendations for future experiments
    5. Any anomalies or interesting patterns observed
    """
    return analysis_prompt


# System prompt for the agent
SYSTEM_PROMPT = """You are an MLflow Experiment Analysis Agent. Your role is to help users analyze machine learning experiments and provide actionable insights.

Your capabilities include:
1. Loading and parsing MLflow experiments from JSON files
2. Querying MLflow tracking servers for experiment data
3. Analyzing experiment performance and providing recommendations
4. Identifying trends, patterns, and anomalies in experimental results

When analyzing experiments, focus on:
- Model performance metrics and their trends
- Parameter sensitivity and optimization opportunities
- Experiment efficiency and resource utilization
- Reproducibility and consistency of results
- Actionable recommendations for improving model performance

Always provide clear, concise, and actionable insights based on the experimental data.
Use the available tools to gather and format experiment data before providing analysis.

Remember to:
- Be objective in your analysis
- Highlight both positive and negative findings
- Suggest concrete next steps
- Consider statistical significance when making claims
- Explain technical concepts clearly for different audiences

You have access to these tools:
- load_mlflow_experiments_from_json: Load experiments from JSON files
- query_mlflow_experiment: Query MLflow tracking server
- analyze_experiment_performance: Analyze performance data

To use a tool, use this format:
Action: tool_name
Action Input: input_parameters

Then wait for the observation before proceeding.
"""


class MLflowAnalysisAgent:
    """
    A LangChain agent for analyzing MLflow experiments using Hugging Face transformers.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-1.5B-Instruct",
        max_length: int = 4096,
        temperature: float = 0.1,
        device: str = "auto",
    ):
        """
        Initialize the MLflow Analysis Agent with Hugging Face transformers.

        Args:
            model_name: Hugging Face model name (default: Qwen2-1.5B for CPU efficiency)
            max_length: Maximum sequence length for generation
            temperature: Temperature for text generation (0.0 to 1.0)
            device: Device to run on ("cpu", "cuda", or "auto")
        """
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.device = self._get_device(device)
        self.llm = None
        self.agent = None
        self.agent_executor = None
        self._setup_agent()

    def _get_device(self, device: str) -> str:
        """Determine the appropriate device to use."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _setup_agent(self):
        """Setup the LangChain agent with Hugging Face model."""
        print(f"Loading model {self.model_name} on {self.device}...")

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device if self.device == "cuda" else None,
            trust_remote_code=True,
        )

        # Create pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=self.max_length,
            temperature=self.temperature,
            do_sample=True,
            device=0 if self.device == "cuda" else -1,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Create LangChain wrapper
        self.llm = HuggingFacePipeline(pipeline=pipe)

        # Define available tools
        tools = [
            load_mlflow_experiments_from_json,
            query_mlflow_experiment,
            analyze_experiment_performance,
        ]

        # Create ReAct prompt template
        template = """Answer the following questions as best you can. You have access to the following tools:

            {tools}

            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Begin!

            {agent_scratchpad}

            Question: {input}
            Thought:"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["input", "agent_scratchpad"],
            partial_variables={
                "tools": "\n".join(
                    [f"{tool.name}: {tool.description}" for tool in tools]
                ),
                "tool_names": ", ".join([tool.name for tool in tools]),
            },
        )

        # Create agent
        self.agent = create_react_agent(llm=self.llm, tools=tools, prompt=prompt)

        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            early_stopping_method="generate",
        )

        print("Agent setup complete!")

    def analyze_from_json(self, json_file_path: str) -> str:
        """
        Analyze MLflow experiments from a JSON file.

        Args:
            json_file_path: Path to JSON file containing experiment data

        Returns:
            Analysis results as string
        """
        query = f"""
        Please load the MLflow experiments data from the JSON file '{json_file_path}' 
        and provide a comprehensive analysis of the experimental results including:
        1. Performance summary
        2. Best runs identification
        3. Parameter impact analysis
        4. Recommendations for improvement
        """

        try:
            result = self.agent_executor.invoke({"input": query})
            return result["output"]
        except Exception as e:
            return f"Error during analysis: {str(e)}"

    def analyze_from_mlflow(self, experiment_id: str, top_n: int = 20) -> str:
        """
        Analyze MLflow experiments directly from MLflow tracking server.

        Args:
            experiment_id: MLflow experiment ID
            top_n: Number of top runs to analyze

        Returns:
            Analysis results as string
        """
        query = f"""
        Please query the MLflow experiment with ID '{experiment_id}' 
        (retrieve top {top_n} runs) and provide a comprehensive analysis 
        of the experimental results including performance trends and recommendations.
        """

        try:
            result = self.agent_executor.invoke({"input": query})
            return result["output"]
        except Exception as e:
            return f"Error during analysis: {str(e)}"

    def custom_analysis(self, query: str) -> str:
        """
        Perform custom analysis based on user query.

        Args:
            query: Custom analysis query

        Returns:
            Analysis results as string
        """
        try:
            result = self.agent_executor.invoke({"input": query})
            return result["output"]
        except Exception as e:
            return f"Error during analysis: {str(e)}"


# Alternative implementation using direct transformers (without LangChain agent)
class DirectTransformerMLflowAnalyzer:
    """
    A direct transformer-based analyzer for MLflow experiments without LangChain agents.
    This approach gives you more control over the generation process.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-1.5B-Instruct",
        max_length: int = 4096,
        device: str = "auto",
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.device = (
            "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        )

        print(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device if self.device == "cuda" else None,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_response(self, prompt: str, max_new_tokens: int = 2048) -> str:
        """Generate response using the transformer model."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length - max_new_tokens,
        )

        if self.device == "cuda":
            inputs = inputs.to("cuda")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return response.strip()

    def analyze_json_file(self, json_file_path: str) -> str:
        """Analyze MLflow experiments from JSON file."""
        try:
            with open(json_file_path, "r") as f:
                data = json.load(f)

            # Create analysis prompt
            prompt = f"""<|im_start|>system
                You are an expert MLflow experiment analyst. Analyze the provided experiment data and give actionable insights.
                <|im_end|>
                <|im_start|>user
                Analyze the following MLflow experiment runs:

                {data}

                Please provide:
                1. Performance summary
                2. Best performing runs
                3. Parameter trends and recommendations
                4. Key insights and next steps
                <|im_end|>
                <|im_start|>assistant
                """

            return self.generate_response(prompt)

        except Exception as e:
            return f"Error analyzing JSON file: {str(e)}"


# Example usage and testing
def main():
    """
    Example usage of both MLflow Analysis implementations.
    """
    print("Choose implementation:")
    print("1. LangChain Agent (full agent with tools)")
    print("2. Direct Transformer (simpler, more direct)")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        print("\nInitializing LangChain-based MLflow Analysis Agent...")
        agent = MLflowAnalysisAgent(
            model_name="Qwen/Qwen2-1.5B-Instruct",  # Lightweight for CPU
            max_length=4096,
            temperature=0.1,
            device="auto",
        )

        # Test with JSON file
        json_file = "mlflow_experiments.json"
        if os.path.exists(json_file):
            print(f"\nAnalyzing {json_file}...")
            result = agent.analyze_from_json(json_file)
            print("Analysis Result:")
            print(result)
        else:
            print(f"JSON file {json_file} not found.")

    elif choice == "2":
        print("\nInitializing Direct Transformer Analyzer...")
        analyzer = DirectTransformerMLflowAnalyzer(
            model_name="Qwen/Qwen2-1.5B-Instruct", device="auto"
        )

        # Test with JSON file
        json_file = "mlflow_experiments.json"
        if os.path.exists(json_file):
            print(f"\nAnalyzing {json_file}...")
            result = analyzer.analyze_json_file(json_file)
            print("Analysis Result:")
            print(result)
        else:
            print(f"JSON file {json_file} not found.")

    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
