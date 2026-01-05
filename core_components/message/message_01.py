import base64
from typing import List, Dict, Any

from langchain_ollama import ChatOllama
from langchain_core.messages import (
    SystemMessage, 
    HumanMessage, 
    AIMessage, 
    ToolMessage, 
    BaseMessage
)
from langchain_core.tools import tool

# ==========================================
# SETUP: Initialize the Model
# ==========================================
model = ChatOllama(
    model="qwen3:14b", 
    temperature=1
)

print(f"--- 1. Simple Text Prompt Invocation ---")
# Ideal for straightforward generation without history
# Ideal for application such that single request, minimal code complexity requirements
response = model.invoke("Write a single sentence haiku about coding.")
print(f"Response: {response.content}\n")


print(f"--- 2. Message Prompts & System Persona ---")
# Using specific message objects to define behavior
# Very useful for identity creation for the LLM application
messages = [
    SystemMessage(content="You are a sarcastic senior Python developer. Be brief."),
    HumanMessage(content="How do I print 'Hello World'?")
]

# Below is also useful for message OpenAI chat models
# messages = [
#     {"role": "system", "content":"You are modern polymath!"},
#     {"role": "user", "content":"How do I manage time!"}
# ]
response = model.invoke(messages)
print(f"Role: {response.type}")
print(f"Content: {response.content}\n")


print(f"--- 3. Message Metadata (Name & ID) ---")
# Adding metadata to track users or trace IDs
human_msg_with_meta = HumanMessage(
    content="What is the capital of France?",
    name="alice_user", 
    id="msg_unique_123"
)
print(f"Created Message with ID: {human_msg_with_meta.id}")
response = model.invoke([human_msg_with_meta])
print(f"Response: {response.content}\n")


print(f"--- 4. Multimodal Content Structure ---")
# Demonstrating the structure for Image/Text inputs.
# Note: Ensure the underlying Ollama model supports vision (like Llava or Qwen-VL) 
# for this to actually analyze the image, otherwise it may just process the text.
multimodal_message = HumanMessage(content=[
    {"type": "text", "text": "What structure is defined in this message?"},
    # Example generic image URL structure as per docs
    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
])

# We won't invoke this to avoid errors if the specific qwen text model 
# doesn't handle image tokens, but this is the valid code implementation.
print(f"Multimodal Message Content: {multimodal_message.content}\n")


print(f"--- 5. Tool Calling (Full Cycle) ---")

# 5a. Define a tool
@tool
def get_weather(location: str) -> str:
    """Get the current weather at a location."""
    # Mock return value
    return f"Sunny, 25°C in {location}"

# 5b. Bind tool to model
model_with_tools = model.bind_tools([get_weather])

# 5c. Invoke model with a query requiring the tool
query = "What is the weather in Tokyo?"
print(f"User: {query}")
ai_msg_with_tool_call = model_with_tools.invoke(query)

# Check if tool was called
if ai_msg_with_tool_call.tool_calls:
    print(f"AI Tool Call: {ai_msg_with_tool_call.tool_calls}")
    
    # 5d. Create the ToolMessage (The Result)
    # We extract the ID and args from the model's response
    tool_call_data = ai_msg_with_tool_call.tool_calls[0]
    
    # Execute the tool (simulated here using the function)
    tool_result = get_weather.invoke(tool_call_data)
    
    tool_msg = ToolMessage(
        content=str(tool_result),
        tool_call_id=tool_call_data["id"],
        name=tool_call_data["name"]
    )
    print(f"Tool Result Message: {tool_msg}")

    # 5e. Continue conversation (System + User + AI-Tool-Call + Tool-Result)
    final_messages = [
        HumanMessage(query),
        ai_msg_with_tool_call, # The model's request to use a tool
        tool_msg               # The result of the tool
    ]
    
    final_response = model_with_tools.invoke(final_messages)
    print(f"Final Answer: {final_response.content}\n")
else:
    print("Model decided not to call a tool.\n")


print(f"--- 6. Manually Constructing AI Messages (History Injection) ---")
# Useful for restoring state or injecting "fake" history
fake_history = [
    SystemMessage("You are a helpful assistant."),
    HumanMessage("Hi!"),
    AIMessage("Hello! How can I help you?"), # Manually created AI response
    HumanMessage("Multiply 5 by 5.")
]
response = model.invoke(fake_history)
print(f"Response based on injected history: {response.content}\n")


print(f"--- 7. Content Blocks & Usage Metadata ---")
# Inspecting the standardized content blocks and token usage
# (Note: Usage metadata availability depends on the Ollama version and model)

print(f"Raw Content: {response.content}")

# Access standardized content blocks (Lazy parsing)
# If the model returned reasoning or specific blocks, they appear here
if hasattr(response, 'content_blocks'):
    print(f"Content Blocks: {response.content_blocks}")

# Access Usage Metadata (Input/Output tokens)
if response.usage_metadata:
    print(f"Usage Metadata: {response.usage_metadata}")
else:
    print("Usage metadata not provided by this model response.")

print(f"\n--- 8. Streaming ---")
# Handling chunks
full_text = ""
print("Streaming response: ", end="", flush=True)
for chunk in model.stream("Count to 3"):
    print(chunk.content, end="", flush=True)
    full_text += str(chunk.content)
print("\nDone.")