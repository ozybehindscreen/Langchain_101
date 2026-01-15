from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime
from langchain_ollama import ChatOllama
from langgraph.types import Command
from langchain.messages import ToolMessage
from pydantic import BaseModel

# --- FIX 1: Update State to include user_name ---
class CustomState(AgentState):
    user_id: str
    user_name: str | None  # Added this so the agent can store the name

class CustomStateWrite(AgentState):
    user_name: str

class CustomContextWrite(BaseModel):
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
    user_id = runtime.state.get("user_id") # Use .get() for safety
    return "User is John Smith" if user_id == "user_123" else "Unknown user"

@tool
def update_user_info(runtime: ToolRuntime[CustomContextWrite, CustomStateWrite]) -> Command:
    """Look up and update user info."""
    # This reads from the 'context' passed in agent.invoke(..., context=...)
    user_id = runtime.context.user_id 
    
    name = "Johan Liebert" if user_id == "user_511" else "Monster"
    
    return Command(update={
        "user_name": name,
        "messages": [
            ToolMessage(
                f"Successfully updated user name to {name}",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })

@tool
def greet(
    runtime: ToolRuntime[CustomContextWrite, CustomStateWrite]
) -> str | Command:
    """Use this to greet the user once you found their info/ID."""
    # This reads from the State (memory)
    user_name = runtime.state.get("user_name", None)
    
    if user_name is None:
        return Command(update={
            "messages": [
                ToolMessage(
                    "Please call the 'update_user_info' tool first to get the user's name.",
                    tool_call_id=runtime.tool_call_id
                )
            ]
        })
    return f"Hello {user_name}"

llm = ChatOllama(model="qwen3:14b", temperature=0)

# --- FIX 2: Pass ALL tools to the agent ---
tools_list = [get_user_info, update_user_info, greet]

agent = create_agent(
    model=llm,
    tools=tools_list, 
    state_schema=CustomState,
)

print("==============Read===============")
result = agent.invoke({
    "messages": "look up user information",
    "user_id": "user_123"
})
print(result["messages"][-1].content)

print("==============Write===============")
# The agent will now:
# 1. Try to greet -> Fail (no name)
# 2. Call update_user_info (using context user_511) -> Update state
# 3. Call greet -> Success
result = agent.invoke(
    {"messages": [{"role": "user", "content": "greet the user"}]},
    context=CustomContextWrite(user_id="user_511"),
)
print(result["messages"][-1].content)