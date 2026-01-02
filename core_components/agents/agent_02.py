from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.tools import tool

# 1. Tool definition
@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers together."""
    return a * b

@tool
def get_weather(city: str) -> str:
    """Provides the weather for a specific city."""
    return f"The weather in {city} is currently 22°C and sunny."

tools = [multiply, get_weather]

# 2. Initialize the model
llm = ChatOllama(
    model="qwen3:14b",
    temperature=0,
    reasoning=False 
)

# 3. Use the high-level create_agent 
# In v1.0, you can pass a system_prompt directly.
agent = create_agent(
    llm, 
    tools, 
    system_prompt="You are a helpful assistant. Use tools when necessary."
)

# 4. Running the agent (The "What to do" part)
# Since the agent IS the executor, we just stream it.
inputs = {"messages": [("user", "What is 134 times 12, and how is the weather in Tokyo? Give answer in 500 words!")]}

for chunk in agent.stream(inputs, stream_mode="updates"): # updates, message and custom are three modes for streaming in the code!
    for node_name, output in chunk.items():
        print(f"--- Node: {node_name} ---")
        # In modern LangChain, the 'model' node outputs the AI's thoughts/calls
        # and the 'tools' node outputs the results.
        if "messages" in output:
            last_message = output["messages"][-1]
            print(last_message.content)