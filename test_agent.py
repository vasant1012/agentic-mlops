from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

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
tools = []

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
)

# Test
result = agent_executor.invoke({"input": "Hello, summarize MLflow run logs"})
print(result["output"])
