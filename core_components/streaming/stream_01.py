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
# for chunk in agent.stream(  
#     {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
#     stream_mode="updates", # This mode yields updates whenever a node in the agent's graph finishes.
# ):
#     # 'chunk' represents a snapshot of the current state of the agent's execution.
#     for step, data in chunk.items():
#         print(f"\n--- step: {step} ---")
        
#         # Access the last message generated in this specific step.
#         # This will usually be either a "Tool Call" (AI) or a "Tool Result" (System).
#         last_msg = data['messages'][-1]
        
#         # .content_blocks contains the structured data (like the tool name and arguments).
#         if hasattr(last_msg, 'content_blocks'):
#             print(f"content: {last_msg.content_blocks}")
#         else:
#             # Fallback for simple string responses (the final answer).
#             print(f"content: {last_msg.content}")

# 5. Execute the Agent using 'stream message mode'
# Streaming mode messages allows you to see the output of the agent streaming tool calls and the final response.

for token, metadata in agent.stream(  
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="messages", # Streams tuples of (token, metadata) from any graph nodes where an LLM is invoked.
):
    print(f"node: {metadata['langgraph_node']}")
    print(f"content: {token}")
    print("\n")


# 6. Customer updates

from langgraph.config import get_stream_writer

def get_star_sign(number_of_stars: str) -> str:
    """Given number of star tell the potential star sign"""
    writer = get_stream_writer()
    # This data is what appears in 'stream_mode="custom"'
    writer(f"Looking up data for stars") 
    writer(f"Acquired data for {number_of_stars}")
    return f"It's Aquarius"

agent_custom = create_agent(llm, tools=[get_star_sign])

print("\n--- RUNNING SECTION 6 ---")
# Prompt must be specific enough to trigger the tool
for chunk in agent_custom.stream(
    {"messages": [{"role": "user", "content": "What is the star sign for 12 stars?"}]},
    stream_mode="custom"
):
    print(f"Custom Chunk: {chunk}")