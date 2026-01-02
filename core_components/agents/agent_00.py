from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

def explain_like_5(topic: str):
    # 1. Define the prompt template: Basically formatted string with placeholders
    prompt = ChatPromptTemplate.from_template(
        "Explain the following topic like I'm 5 years old: {topic}"
    )
    # 2. Create the LLM instance, using the Ollama model for langchain learning, because of BFCL benchmark which allows tool/function calling
    llm = ChatOllama(model="qwen3:14b", temperature=0)
    chain = prompt | llm # Pipe the prompt into the LLM to create a chain this is basically saying model.invoke(prompt.format(topic="coding"))
    # response = chain.invoke({"topic": topic})

    # 3. Return the response content
    for chunk in chain.stream({"topic": topic}):
        # 2. Extract the content and print it immediately
        # end="" prevents a new line, flush=True ensures it hits the screen instantly
        yield chunk.content

if __name__ == "__main__":
    print(explain_like_5("Quantum Computing"))