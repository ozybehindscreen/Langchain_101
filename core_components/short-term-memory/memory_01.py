from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

# Memory is import to understant the context of conversation that we are meaning to have
# Mostly because of Large Conversation, LLM tends to start "forgetting things"
# This is where Langgraph provide memory storage methods for RAG
llm = ChatOllama(model="qwen3:14b")

agent = create_agent(model = llm, checkpointer=InMemorySaver()) #This InMemorySaver allows use to store memory thread wise in our configurations!

output = agent.invoke(
    {"messages": [{"role": "user", "content": "Hi! My name is Bob."}]},
    {"configurable": {"thread_id": "1"}},  
)

output2 = agent.invoke(
    {"messages": [{"role": "user", "content": "Can you tell me more above Taiwan!"}]},
    {"configurable": {"thread_id": "1"}},  
)

output3 = agent.invoke(
    {"messages": [{"role": "user", "content": "Can you tell me export and Imports of that country?"}]},
    {"configurable": {"thread_id": "1"}},  
)

print("--- Full State ---")
print(output, output2, output3)

print("--Messages--")
print("First\n",output["messages"][-1].content)
print("Second\n",output2["messages"][-1].content)
print("Third\n",output3["messages"][-1].content)