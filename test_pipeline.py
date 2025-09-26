from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langcahin_wrappers import make_tools
from logger import logger
from mlflow import MlflowClient


# Load Qwen2 model + tokenizer (CPU only, fp16 disabled)
model_id = "Qwen/Qwen2-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu")

# HuggingFace pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.2,
)

# Wrap in LangChain LLM
llm = HuggingFacePipeline(pipeline=pipe)

# Load tools (replace with your MLflow log parser tool)
tools = make_tools(llm)

# Pull a ReAct prompt from LangChain hub
prompt = hub.pull("hwchase17/react")

# Create the agent
agent = create_react_agent(
    llm=llm,      # ✅ not a string, but a LangChain LLM
    tools=tools,
    prompt=prompt,
)

# Create an executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True,
                        help="MLflow experiment id")
    parser.add_argument("--action", default="summarize",
                        choices=["summarize", "recommend", "raw"])
    args = parser.parse_args()

    client = MlflowClient()
    experiment_id = client.get_experiment_by_name(
        args.experiment_name)._experiment_id

    # Test
    if args.action == "summarize":
        out = agent_executor.invoke(
            {"input": f"summarize_runs experiment_id={experiment_id}"})
        logger.info(out['output'])
    elif args.action == "recommend":
        out = agent_executor.invoke({"input":
                                     f"recommend_hparams experiment_id={experiment_id}"})  # NOQA E501
        logger.info(out['output'])
    elif args.action == "raw":
        out = agent_executor.invoke({"input":
                                     f"get_runs_summary experiment_id={experiment_id}"})  # NOQA E501
        logger.info(out['output'])
