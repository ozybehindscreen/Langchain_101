from typing import Annotated, Sequence, TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver

# 1. Initialize the Qwen3:14b model via Ollama
llm = ChatOllama(model="qwen3:14b", temperature=0)

# 2. Define the Trimming Logic
# This function acts as your 'middleware' to manage the context window
def trim_messages(state: MessagesState):
    messages = state["messages"]
    
    # We want to keep the context window small (e.g., last 3 messages)
    # But we usually want to keep the very first message if it's a System prompt
    if len(messages) <= 5:
        return {"messages": []} # No deletion needed yet

    # Identify messages to delete (everything except the last 3)
    # We exclude the first message if you want to keep a persistent system instruction
    delete_cmds = [RemoveMessage(id=m.id) for m in messages[:-5]]
    
    return {"messages": delete_cmds}

# 3. Define the Agent/Model node
def call_model(state: MessagesState):
    # Before calling the model, we could trigger trimming, 
    # but in LangGraph, we can also do it as a separate node.
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# 4. Build the Graph
workflow = StateGraph(MessagesState)

# Add our nodes
workflow.add_node("trimmer", trim_messages)
workflow.add_node("agent", call_model)

# Set the flow: Start -> Trim -> Agent -> End
workflow.add_edge(START, "trimmer")
workflow.add_edge("trimmer", "agent")
workflow.add_edge("agent", END)

# Add memory for thread persistence
checkpointer = InMemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# 5. Execution
config = {"configurable": {"thread_id": "qwen_chat_1"}}

def chat(text: str):
    print(f"\n[User]: {text}")
    input_msg = {"messages": [HumanMessage(content=text)]}
    result = app.invoke(input_msg, config)
    last_msg = result["messages"][-1]
    print(f"[AI]: {last_msg.content}")

# Running the conversation sequence
chat("hi, my name is bob")
chat("write a short poem about cats")
chat("now do the same but for dogs")
chat("what's my name?")