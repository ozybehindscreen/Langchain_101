from langchain.agents import create_agent
from langchain_ollama import ChatOllama

# 1. Define the "Tool" - This is a Python function the LLM can decide to call.
# The docstring is crucial; the LLM uses it to understand WHEN and HOW to use the function.
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    # In a real app, you would call an API like OpenWeatherMap here.
    return f"It's always sunny in {city}!"

# 2. Initialize the Local LLM via Ollama. 
# Ensure 'qwen3:14b' is pulled and running in your local Ollama instance.
llm = ChatOllama(model="qwen3:14b")

# 3. Create the Agent.
# This binds the LLM and the list of tools together into a runnable execution unit.
agent = create_agent(
    model=llm,
    tools=[get_weather],
)

# 4. Execute the Agent using 'stream' mode.
# Streaming allows you to see the intermediate steps (thoughts/tool calls) as they happen.
for chunk in agent.stream(  
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="updates", # This mode yields updates whenever a node in the agent's graph finishes.
):
    # 'chunk' represents a snapshot of the current state of the agent's execution.
    for step, data in chunk.items():
        print(f"\n--- step: {step} ---")
        
        # Access the last message generated in this specific step.
        # This will usually be either a "Tool Call" (AI) or a "Tool Result" (System).
        last_msg = data['messages'][-1]
        
        # .content_blocks contains the structured data (like the tool name and arguments).
        if hasattr(last_msg, 'content_blocks'):
            print(f"content: {last_msg.content_blocks}")
        else:
            # Fallback for simple string responses (the final answer).
            print(f"content: {last_msg.content}")