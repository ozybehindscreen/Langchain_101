import operator
from typing import Annotated, List, Literal, TypedDict, Union

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

# --- 1. Define the State ---
# This holds the conversation history
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# --- 2. Setup Models ---
# The main agent that generates the content
# (Using a smaller model for the chat, e.g., llama3.2)
main_model = ChatOllama(model="llama3.2", temperature=0.7)

# The safety guardrail model
# (Using the model you specified)
safety_model = ChatOllama(model="qwen3:14b", temperature=0)

# --- 3. Define the Structured Output Schema ---
class ResponseSafety(BaseModel):
    """Evaluate a response as safe or unsafe."""
    evaluation: Literal["safe", "unsafe"] = Field(
        description="Classify the response. 'unsafe' includes hate speech, dangerous instructions, or PII."
    )

# --- 4. Define Graph Nodes ---

def agent_node(state: AgentState):
    """The main chatbot node."""
    messages = state["messages"]
    response = main_model.invoke(messages)
    return {"messages": [response]}

def safety_guardrail_node(state: AgentState):
    """
    Model-based guardrail: Use an LLM to evaluate response safety.
    Adapted from your snippet to work in a standard node.
    """
    messages = state["messages"]
    if not messages:
        return None

    last_message = messages[-1]
    
    # Only check AI messages
    if not isinstance(last_message, AIMessage):
        return None

    print(f"\n--- Checking Safety of: '{last_message.content[:50]}...' ---")

    # Use .with_structured_output to force the model to use the Pydantic schema
    evaluator = safety_model.with_structured_output(ResponseSafety)
    
    result = evaluator.invoke([
        {
            "role": "system",
            "content": "Evaluate this AI response as generally safe or unsafe."
        },
        {
            "role": "user",
            "content": f"AI response: {last_message.content}"
        }
    ])

    print(f"--- Verdict: {result.evaluation} ---")

    if result.evaluation == "unsafe":
        # In LangGraph, to "rewrite" the last message, we can 
        # return a new message that acts as the sanitized version.
        # Alternatively, we could raise an error.
        sanitized_content = "I cannot provide that response. Please rephrase your request."
        
        # We replace the content of the last message locally for display,
        # or append a correction. Here we verify logic by printing.
        # To strictly overwrite in the graph history, specialized reducers are needed,
        # but for this example, we will return a system alert or replacement.
        return {"messages": [AIMessage(content=sanitized_content)]}

    # If safe, return nothing (state remains same)
    return None

# --- 5. Build the Graph ---

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", agent_node)
workflow.add_node("safety_check", safety_guardrail_node)

# Add edges
workflow.add_edge(START, "agent")
workflow.add_edge("agent", "safety_check")
workflow.add_edge("safety_check", END)

# Compile
app = workflow.compile()

# --- 6. Execution (Main Loop) ---

if __name__ == "__main__":
    print("Initialize Safety Graph... (Type 'quit' to exit)")
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["quit", "exit"]:
            break
            
        initial_state = {"messages": [HumanMessage(content=user_input)]}
        
        # Run the graph
        # We explicitly filter for the final output
        events = app.stream(initial_state, stream_mode="values")
        
        final_message = None
        for event in events:
            if "messages" in event:
                final_message = event["messages"][-1]
        
        if final_message:
            print(f"Assistant: {final_message.content}")