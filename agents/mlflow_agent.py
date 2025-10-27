from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
import torch
from langchain_core.globals import set_debug, set_verbose
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tools.analysis_tool import analyze_experiment_performance, read_mlflow_logs  # NOQA E501


class MLflowAnalysisAgent:
    set_debug(True)
    set_verbose(True)

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-1.5B-Instruct",
        max_length: int = 4096,
        temperature: float = 0.2,
        device: str = "auto"
    ):
        self.model_name = model_name
        self.device = self._get_device(device)
        self.temperature = temperature
        self.max_length = max_length
        self.temperature = temperature
        self._setup_agent()

    def _get_device(self, device: str) -> str:
        """Determine the appropriate device to use."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _setup_agent(self):
        """Setup the LangChain agent using Hugging Face API instead of local model."""  # NOQA E501
        print(f"Loading model {self.model_name} on {self.device}...")  # NOQA E501

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
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

        # ✅ Register tools
        tools = [
            read_mlflow_logs,
            analyze_experiment_performance,
        ]

        # ✅ ReAct prompt template
        template = """
        You are an ML analysis expert working with MLflow experiments.
        You have access to the following tools:
        {tools}

        Follow this format strictly:
        Question: the user's input question
        Thought: your reasoning
        Final Answer: provide the final JSON output with keys:
        {{
          "summary": "...",
          "best_runs": [...],
          "key_metrics": {{...}},
          "recommendations": "..."
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
                "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in tools]),  # NOQA E501
                "tool_names": ", ".join([tool.name for tool in tools]),
            },
        )

        self.agent = create_react_agent(
            llm=self.llm, tools=tools, prompt=prompt)

        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=1,
            early_stopping_method="generate",
        )

        print("Agent setup complete using HF API.")

    def analyze_from_json(self, json_file_path: str):
        query = f"Read and summarize MLflow experiments from {json_file_path}."
        result = self.agent_executor.invoke({"input": query})
        return result["output"]
