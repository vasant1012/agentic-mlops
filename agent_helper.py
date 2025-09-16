from langchain.agents import create_react_agent, AgentExecutor
from local_hugging_face import LocalHFAdapter
from langcahin_wrappers import make_tools
# used only as a type/sample; we will supply our custom LLM wrapper
from langchain import hub


def build_agent(llm_adapter: LocalHFAdapter):
    tools = make_tools(llm_adapter)
    # We use LLmAdapter as the LLM to the agent
    # Load prompt template from LangChain hub (react.json for ReAct agent)
    prompt = hub.pull("hwchase17/react")

    # Create the agent
    agent = create_react_agent(
        llm=llm_adapter,   # your HuggingFace Qwen adapter
        tools=tools,
        prompt=prompt,
    )

    # Wrap with AgentExecutor
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    return agent_executor
