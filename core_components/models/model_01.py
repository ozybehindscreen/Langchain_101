import os
from typing import List

from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

# ---------------------------------------------------
# Basic init: local Qwen3:14b via Ollama
# ---------------------------------------------------

# If you run Ollama on another host/port, set:
# os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"

model = ChatOllama(
    # Provider-style name: "ollama/qwen3:14b"
    # (LangChain resolves this through the Ollama integration)
    model = "qwen3:14b",
    temperature=1,
    logprobs=True,
)

# ---------------------------------------------------
# 1) Simple invoke (single turn)
# ---------------------------------------------------

def simple_invoke():
    resp = model.invoke("Give me 3 bullet points on LangChain use cases.")
    print("\n=== INVOKE ===")
    print(resp.content)


# ---------------------------------------------------
# 2) Invoke with full message history
# ---------------------------------------------------

def messages_invoke():
    messages = [
        SystemMessage(
            content="You are a concise assistant for Python and LangChain questions."
        ),
        HumanMessage(content="Explain what LangChain chat models are in 2 sentences."),
        HumanMessage(
            content="Now give 2 concrete examples of using them with local models."
        ),
    ]
    # Will learn more about messages and their info in message directory
    resp = model.invoke(messages)
    print("\n=== MESSAGES INVOKE ===")
    print(resp.content)


# ---------------------------------------------------
# 3) Streaming
# ---------------------------------------------------

def streaming_example():
    # Most Modern LLM work with this
    print("\n=== STREAM ===")
    for chunk in model.stream("Explain streaming responses in LangChain in 3 sentences."):
        print(chunk.content, end="", flush=True)
    print()


# ---------------------------------------------------
# 4) Batch
# ---------------------------------------------------

def batch_example():

    print("\n=== BATCH ===")
    prompts = [
        "What is LangChain?",
        "What is Ollama?",
        "How do they work together for local LLMs?",
    ]
    responses = model.batch(prompts)
    for i, r in enumerate(responses):
        print(f"\n--- Answer {i+1} ---")
        print(r.content)


# ---------------------------------------------------
# 5) Tool calling
# ---------------------------------------------------

@tool
def get_weather(location: str) -> str:
    """Get the weather at a location (dummy implementation)."""
    # In real code, you’d call a weather API here.
    return f"It's sunny in {location} with 27°C."

def tool_calling_example():
    print("\n=== TOOL CALLING ===")
    model_with_tools = model.bind_tools([get_weather])

    # Step 1: model decides whether to call tools
    messages = [HumanMessage(content="What's the weather in Mumbai right now?")]
    ai_msg = model_with_tools.invoke(messages)
    messages.append(ai_msg)

    # Step 2: execute any tools requested
    for tool_call in ai_msg.tool_calls:
        if tool_call["name"] == "get_weather":
            tool_result = get_weather.invoke(tool_call)
            messages.append(tool_result)

    # Step 3: send tool results back for final answer
    final_resp = model_with_tools.invoke(messages)
    print(final_resp.content)


# ---------------------------------------------------
# 6) Structured output with Pydantic
# ---------------------------------------------------

class Movie(BaseModel):
    """A movie with details."""
    title: str = Field(..., description="The title of the movie")
    year: int = Field(..., description="The year the movie was released")
    director: str = Field(..., description="The director of the movie")
    rating: float = Field(..., description="The movie's rating out of 10")

def structured_output_example():
    print("\n=== STRUCTURED OUTPUT ===")
    model_with_structure = model.with_structured_output(Movie)
    movie: Movie = model_with_structure.invoke(
        "Provide details about the movie Inception."
    )
    print(movie)
    print("Title:", movie.title)
    print("Year:", movie.year)
    print("Director:", movie.director)
    print("Rating:", movie.rating)


# ---------------------------------------------------
# 7) “Reasoning effort” config (model-dependent)
#    Here we just show how to pass extra kwargs.
# ---------------------------------------------------

def reasoning_style_example():
    print("\n=== REASONING STYLE (CONFIG) ===")
    # Some providers support explicit reasoning params; for a local
    # Ollama model you usually just tweak temperature / max_tokens etc.
    # This shows using a different config on a per-call basis.
    resp = model.invoke(
        "Explain step by step how LangChain tool calling works.",
        config={
            "run_name": "reasoning_demo",
            "tags": ["reasoning", "example"],
            "metadata": {"mode": "step_by_step"},
        },
    )
    print(resp.content)


# ---------------------------------------------------
# 8) Multimodality placeholder (text-only here)
#    Qwen3 in Ollama is typically text-only; showing the API pattern.
# ---------------------------------------------------

def multimodal_placeholder():
    print("\n=== MULTIMODAL PLACEHOLDER ===")
    # If you had an image-capable model, you could send content blocks.
    # For Qwen3 text via Ollama, treat it as normal text.
    resp = model.invoke(
        [
            SystemMessage("You are an assistant that describes images."),
            HumanMessage("Imagine there is a diagram of LangChain + Ollama architecture. Describe it."),
        ]
    )
    print(resp.content)


# ---------------------------------------------------
# 9) Metrtics on LLMs and it's Output (model metrics)
#    Metrics such as model.profile, logprobs, token usuage, 
# ---------------------------------------------------

def model_metrics():
    import pprint
    print("\n=== MODEL METRICS ===")
    
    # 1. Standard Invoke to get a response object
    # We use a simple prompt to generate enough tokens for meaningful metrics
    resp = model.invoke("Write a 5-sentence story about a robot learning to paint.")
    print(f"Response \n - Response: {resp.content}")
    pprint.pprint(resp.response_metadata)
    print(f" - LogProbs: {resp.response_metadata.get('logprobs')}")
    # 2. Access Standardized LangChain Metrics
    # These work across almost all providers (OpenAI, Anthropic, Ollama, etc.)
    usage = resp.usage_metadata
    print(f"Standardized Usage:")
    print(f" - Prompt Tokens: {usage.get('input_tokens')}")
    print(f" - Completion Tokens: {usage.get('output_tokens')}")
    print(f" - Total Tokens: {usage.get('total_tokens')}")

    # 3. Access Ollama-Specific Metrics (The 'Good Stuff')
    # These are found in response_metadata and are unique to Ollama's local engine
    meta = resp.response_metadata
    print(f"\nOllama Performance Metrics:")
    print(f" - Model: {meta.get('model')}")
    print(f" - Total Duration: {meta.get('total_duration') / 1e9:.2f}s")  # ns to s
    print(f" - Load Duration: {meta.get('load_duration') / 1e9:.4f}s")
    
    # Eval Rate is essentially your Tokens Per Second (TPS)
    eval_rate = meta.get('eval_count', 0) / (meta.get('eval_duration', 1) / 1e9)
    print(f" - Generation Speed: {eval_rate:.2f} tokens/s")
    
    # 4. View Raw Metadata for Debugging
    # print("\nFull Metadata Dictionary:")
    # import pprint
    # pprint.pprint(meta)


# ---------------------------------------------------
# 9) Main
# ---------------------------------------------------

if __name__ == "__main__":
    # simple_invoke()
    # messages_invoke()
    # streaming_example()
    # batch_example()
    # tool_calling_example()
    # structured_output_example()
    # reasoning_style_example()
    # multimodal_placeholder()
    model_metrics()
