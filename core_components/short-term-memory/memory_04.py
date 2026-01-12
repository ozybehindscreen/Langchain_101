from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig

# 1. Initialize your local Ollama models
# The main agent model
llm = ChatOllama(model="qwen3:14b", temperature=0)

# A smaller model for summarization to save resources
summarizer_llm = ChatOllama(model="qwen3:1.7b", temperature=0)

checkpointer = InMemorySaver()

# 2. Create the agent with local LLMs
agent = create_agent(
    model=llm,
    tools=[],
    middleware=[
        # More in Middleware module
        SummarizationMiddleware(
            model=summarizer_llm,
            trigger=("tokens", 4000),
            keep=("messages", 20)
        )
    ],
    checkpointer=checkpointer,
)

# 3. Execution
config: RunnableConfig = {"configurable": {"thread_id": "1"}}

agent.invoke({"messages": "hi, my name is bob"}, config)
agent.invoke({"messages": "write a short poem about cats"}, config)
agent.invoke({"messages": "now do the same but for dogs"}, config)
final_response = agent.invoke({"messages": "what's my name?"}, config)

final_response["messages"][-1].pretty_print()