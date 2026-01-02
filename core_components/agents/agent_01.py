from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# 1. Define your models
simple_model = ChatOllama(model="llama3.2", temperature=0)
complex_model = ChatOllama(model="qwen3:14b", temperature=0)

# 2. Define the Routing Logic
def route_model(info):
    # 'info' will be the output from the prompt (the formatted messages)
    # We inspect the last message content to decide
    last_message = info.messages[-1].content.lower()
    
    print(f"\n[DEBUG] Routing based on: '{last_message[:20]}...'")
    
    if "search" in last_message or "trending" in last_message:
        print("[DEBUG] Selected: Complex Model (Qwen3)")
        return complex_model
    else:
        print("[DEBUG] Selected: Simple Model (Llama3.2)")
        return simple_model

# 3. Create the Prompt
# Note: Added {input} so the user query is actually injected
prompt = ChatPromptTemplate.from_template(
    "Answer the following question concisely: {input}"
)

# 4. Create the Chain
# prompt -> determines which model to use -> calls that model -> parses string
chain = prompt | RunnableLambda(route_model) | StrOutputParser()

# Import Note: RunnableLambda allows us to insert custom logic (like routing) into the chain
# Runnable is a fundamental building block in langchain core that represents a single task or operations, it allows easily composable components into complex workflows
# using LangChain Expression Language (LCEL) we can define how data flows through these runnables

def model_call(user_input: str):
    # Stream the output
    for chunk in chain.stream({"input": user_input}):
        print(chunk, end="", flush=True)

if __name__ == "__main__":
    # This contains "trending", so it should trigger the Complex Model
    model_call("Search, what is trending in education sector nowadays?")