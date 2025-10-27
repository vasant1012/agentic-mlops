from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain_core.globals import set_debug, set_verbose
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
# from tools.mlflow_tools import load_mlflow_experiments_from_json, query_mlflow_experiment  # NOQA E501
from tools.analysis_tool import analyze_experiment_performance  # read_mlflow_logs # NOQA E501


class MLflowAnalysisAgent:
    """
    A LangChain agent for analyzing MLflow experiments using Hugging Face transformers. # NOQA E501
    """
    set_debug(True)
    set_verbose(True)

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
            model_name: Hugging Face model name (default: Qwen2-1.5B for CPU efficiency) # NOQA E501
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
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,  # NOQA E501
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
            # read_mlflow_logs,
            # load_mlflow_experiments_from_json,
            # query_mlflow_experiment,
            analyze_experiment_performance,
        ]

        # Create ReAct prompt template
        template = """Answer the following questions as best you can.
            You have access to the following tools:

            {tools}

            Use the following format exactly:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            (this Thought/Action/Action Input/Observation can repeat 2 times)
            Thought: I now know the final answer
            Final Answer: respond strictly as a JSON object
            with this structure:
            {{
            "summary": "text summary of what you analyzed",
            "best_run": {{
                "run_id": "id or description of best run",
                "metrics": {{ "metric_name": value }},
                "params": {{ "param_name": value }}
            }},
            "insights": [
                "key finding 1",
                "key finding 2",
                "key finding 3"
            ]
            }}

            IMPORTANT: Do not include any text outside the JSON object.

            Begin reasoning now.
            (IMPORTANT: Only produce 'Final Answer' at the end in JSON format.
              Do NOT include Action or Observation steps.)
            {agent_scratchpad}

            Question: {input}
            Thought:
            """

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

        print("Prompt input variables:", prompt.input_variables)

        # Create agent
        self.agent = create_react_agent(
            llm=self.llm, tools=tools, prompt=prompt)

        # prompt_text = prompt.format(
        #     input="Read mlflow_experiments.json and summarize metrics.",
        #     agent_scratchpad=""
        # )
        # print(prompt_text)
        # response = self.llm.invoke(prompt_text)
        # print("LLM raw output:", response)

        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=1,
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
        Please load the MLflow experiments data from the JSON file '{json_file_path}' # NOQA E501
        and provide a comprehensive analysis of the experimental results including:
        1. Performance summary
        2. Best runs identification
        3. Parameter impact analysis
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
        of the experimental results including performance trends and recommendations. # NOQA E501
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
