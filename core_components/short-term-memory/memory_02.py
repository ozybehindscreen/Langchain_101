import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

# 1. Load environment variables
load_dotenv()

# Ensure DATABASE_URL is set in your .env file
# Example: DATABASE_URL=postgresql://postgres:password@localhost:5432/postgres
DB_URI = os.getenv("DATABASE_URL")

if not DB_URI:
    raise ValueError("DATABASE_URL not found in environment variables")

def test_memory():
    # 2. Setup the Postgres Checkpointer
    # from_conn_string manages the connection pool context for us
    print(f"Connecting to database...")
    
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        # Initialize the database tables if they don't exist
        checkpointer.setup()
        
        # 3. Initialize the LLM
        # Make sure this model is pulled in Ollama (e.g., `ollama pull qwen2.5:14b`)
        llm = ChatOllama(model="qwen3:14b") 

        # 4. Create the Agent (Graph)
        # We pass an empty list of tools for a simple chat test, 
        # but create_react_agent expects a list (even if empty)
        agent_graph = create_react_agent(
            model=llm,
            tools=[], 
            checkpointer=checkpointer
        )

        # 5. Define a specific thread ID. 
        # This ID is the key to retrieving memory from Postgres.
        thread_config = {"configurable": {"thread_id": "memory_test_thread_001"}}

        print("\n--- Test 1: Teaching the Agent ---")
        user_input_1 = "Hi! My name is Alice. Please remember this."
        print(f"User: {user_input_1}")
        
        # Invoke the agent
        response_1 = agent_graph.invoke(
            {"messages": [HumanMessage(content=user_input_1)]},
            config=thread_config
        )
        
        # Print Agent Response
        print(f"Agent: {response_1['messages'][-1].content}")

        print("\n--- Test 2: Verifying Memory ---")
        # In a real app, this could happen days later or after a server restart.
        # As long as we use the same 'thread_id' and the same DB, it should remember.
        user_input_2 = "What is my name?"
        print(f"User: {user_input_2}")
        
        response_2 = agent_graph.invoke(
            {"messages": [HumanMessage(content=user_input_2)]},
            config=thread_config
        )
        
        answer = response_2['messages'][-1].content
        print(f"Agent: {answer}")

        # Simple assertion
        if "Alice" in answer:
            print("\n✅ SUCCESS: Memory is working! The agent retrieved the name from Postgres.")
        else:
            print("\n❌ FAILURE: The agent did not recall the name.")

if __name__ == "__main__":
    test_memory()