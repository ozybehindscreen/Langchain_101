from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime
from langchain_ollama import ChatOllama

class CustomState(AgentState):
    user_id: str

@tool
def get_user_info(
    runtime: ToolRuntime
) -> str:
    """Look up user info."""
    
    # --- COMMENT: READING SHORT-TERM MEMORY FROM TOOL ---
    # By annotating an argument with `ToolRuntime`, we get access to the 
    # agent's current state (acting as short-term memory).
    # This allows the tool to read context variables (like 'user_id') 
    # that were passed into '.invoke()', without requiring the LLM 
    # to hallucinate or explicitly pass the user_id as a parameter.
    user_id = runtime.state["user_id"]
    
    return "User is John Smith" if user_id == "user_123" else "Unknown user"

llm = ChatOllama(model="qwen3:14b", temperature=0)

agent = create_agent(
    model=llm,
    tools=[get_user_info],
    state_schema=CustomState,
)

result = agent.invoke({
    "messages": "look up user information",
    # This value is stored in state (short-term memory) and read by the tool above
    "user_id": "user_123" 
})
print(result["messages"][-1].content)
# > User is John Smith.