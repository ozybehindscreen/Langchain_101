from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import dynamic_prompt, before_model, ModelRequest
from langchain.tools import tool, ToolRuntime
from langchain_ollama import ChatOllama
from langgraph.types import Command
from langchain.messages import ToolMessage
from typing import TypedDict, Any
from langgraph.runtime import Runtime

# --- SCHEMAS ---

# Context: Read-only configuration passed per invoke
class CustomContext(TypedDict):
    user_id: str

# State: Short-term memory maintained across the thread
class CustomState(AgentState):
    user_name: str | None

# --- MIDDLEWARE ---

@dynamic_prompt
def personalize_system_prompt(request: ModelRequest) -> str:
    """
    Access Context and State to build a dynamic system prompt.
    """
    # 1. Access Context (Configuration) via request.runtime.context
    user_id = request.runtime.context.get("user_id", "Unknown")
    
    # 2. Access State (Short-term Memory) via request.state
    # FIX: Use 'request.state' instead of 'request.runtime.state'
    state = request.state
    user_name = state.get("user_name")

    # Construct the System Prompt
    system_prompt = f"You are a helpful assistant. The User ID is {user_id}."
    
    if user_name:
        system_prompt += f" You are talking to {user_name}."
    else:
        system_prompt += " You do not know the user's name yet."

    return system_prompt

@before_model
def log_and_validate_state(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """
    Access State before the model runs to perform validation or logging.
    Note: @before_model receives 'state' explicitly as an argument.
    """
    user_name = state.get("user_name")
    
    if user_name:
        print(f"[Middleware Log] Current User Name in State: {user_name}")
    else:
        print(f"[Middleware Log] User Name is currently unknown.")
        
    return None # No changes to messages

# --- TOOLS ---

@tool
def update_user_info(
    runtime: ToolRuntime
) -> Command:
    """Look up user info based on context ID and update memory."""
    
    # Accessing the Context via ToolRuntime
    user_id = runtime.context.get("user_id")
    
    # Logic to determine name
    name = "Johan Liebert" if user_id == "user_511" else "Monster"
    
    # Updating State (Short-term memory) via Command
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
    runtime: ToolRuntime
) -> str | Command:
    """Greet the user. Requires user_name to be known."""
    
    # Accessing State via ToolRuntime
    # Note: Tools access state via 'runtime.state', unlike Middleware which uses 'request.state'
    user_name = runtime.state.get("user_name", None)
    
    if user_name is None:
        return Command(update={
            "messages": [
                ToolMessage(
                    "I don't know the user's name yet. Use 'update_user_info' first.",
                    tool_call_id=runtime.tool_call_id
                )
            ]
        })
    return f"Hello, {user_name}! How are you?"

# --- AGENT SETUP ---

llm = ChatOllama(model="qwen3:14b", temperature=0)

agent = create_agent(
    model=llm,
    tools=[update_user_info, greet],
    middleware=[personalize_system_prompt, log_and_validate_state],
    state_schema=CustomState,
    context_schema=CustomContext,
)

# --- EXECUTION ---

print("============== Run 1: Context Only ===============")
# Middleware sees user_name is None -> Prompt: "...You do not know the user's name yet."
result_1 = agent.invoke(
    {"messages": [{"role": "user", "content": "What is my name?"}]},
    context=CustomContext(user_id="user_511")
)
print(f"AI: {result_1['messages'][-1].content}")


print("\n============== Run 2: Context + State Update ===============")
# 1. AI realizes it doesn't know the name.
# 2. Calls update_user_info -> State updates to "Johan Liebert".
# 3. Middleware sees "Johan Liebert" -> Prompt: "...You are talking to Johan Liebert."
result_2 = agent.invoke(
    {"messages": [{"role": "user", "content": "find my info and then greet me"}]},
    context=CustomContext(user_id="user_511")
)
print(f"AI: {result_2['messages'][-1].content}")