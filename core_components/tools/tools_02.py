import logging
from typing import Annotated

from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

# LangGraph imports
from langgraph.prebuilt import create_react_agent, InjectedStore
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore

# 1. Define Context Schema
class UserContext(BaseModel):
    user_id: str
    username: str

# 2. Define LLM
llm = ChatOllama(model="qwen3:14b", temperature=0)

# 3. Tool Accessing Runtime Context
# We use RunnableConfig to access data passed at runtime (like user info)
@tool
def get_current_user_info(config: RunnableConfig) -> str:
    """Get information about the currently logged-in user."""
    # Retrieve the context from the 'configurable' dictionary
    user: UserContext = config["configurable"].get("user_context")
    if not user:
        return "No user context found."
    return f"User ID: {user.user_id}, Username: {user.username}"

# 4. Tool Accessing Store (Long-term Memory)
# We use Annotated[BaseStore, InjectedStore] to let LangGraph inject the store automatically
@tool
def save_user_fact(
    fact: str, 
    config: RunnableConfig, 
    store: Annotated[BaseStore, InjectedStore]
) -> str:
    """Save a fact about the user to long-term memory."""
    user: UserContext = config["configurable"].get("user_context")
    if not user:
        return "Error: No user context."
        
    # Store data keyed by user_id inside the "facts" namespace
    store.put(("facts", user.user_id), "latest_fact", {"text": fact})
    return "Fact saved successfully."

@tool
def recall_user_fact(
    config: RunnableConfig, 
    store: Annotated[BaseStore, InjectedStore]
) -> str:
    """Recall the last saved fact about the user."""
    user: UserContext = config["configurable"].get("user_context")
    if not user:
        return "Error: No user context."

    # Retrieve data
    item = store.get(("facts", user.user_id), "latest_fact")
    return f"Recall: {item.value['text']}" if item else "No facts stored."

# 5. Create Agent with Store
tools = [get_current_user_info, save_user_fact, recall_user_fact]
store = InMemoryStore()

# We use create_react_agent from langgraph.prebuilt
agent = create_react_agent(llm, tools=tools, store=store)

def run_context_demo():
    print("--- Context & Memory Demo with Qwen3:14b ---")

    # Simulate a user session object
    current_user = UserContext(user_id="u_123", username="Alice")

    # IMPORTANT: We pass context via 'configurable'
    # This ensures it is passed cleanly to the tools without serialization errors
    config = {"configurable": {"user_context": current_user}}

    # 1. Test Context Access
    print(f"\nUser: Who am I?")
    resp = agent.invoke(
        {"messages": [HumanMessage(content="Who am I? use the user info tool.")]},
        config=config
    )
    print(f"Agent: {resp['messages'][-1].content}")

    # 2. Test Memory Write
    print(f"\nUser: Remember that I love coding.")
    resp = agent.invoke(
        {"messages": [HumanMessage(content="Remember that I love coding.")]},
        config=config
    )
    print(f"Agent: {resp['messages'][-1].content}")

    # 3. Test Memory Read
    print(f"\nUser: What did I tell you to remember?")
    resp = agent.invoke(
        {"messages": [HumanMessage(content="What did I tell you to remember?")]},
        config=config
    )
    print(f"Agent: {resp['messages'][-1].content}")

if __name__ == "__main__":
    try:
        run_context_demo()
    except Exception as e:
        print(f"Error: {e}")