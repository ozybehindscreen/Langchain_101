import sys
from typing import Literal
from collections import Counter

from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent 
from pydantic import BaseModel, Field

# 1. Define the LLM
llm = ChatOllama(model="qwen3:14b", temperature=0)

# 2. Basic Tool Definition
@tool
def calculate_area(length: float, width: float) -> str:
    """Calculates the area of a rectangle given length and width."""
    return f"The area is {length * width} square units."

# 2.5 Customize tool properties (Fixed)
# We use 'string_counter' (underscores) to ensure the LLM handles the function name correctly.
@tool("string_counter", description="Calculates number of characters in a string!")
def calc(word: str) -> str:
    """Calculates number of characters in a string!"""
    # Simply return the count of characters
    return str(dict(Counter(word)))

# 3. Advanced Tool Definition using Pydantic
class WeatherInput(BaseModel):
    city: str = Field(description="The name of the city to get weather for")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius", 
        description="Temperature unit preference"
    )

@tool(args_schema=WeatherInput)
def get_weather(city: str, units: str = "celsius") -> str:
    """Get the current weather for a specific city."""
    temp = 25 if units == "celsius" else 77
    return f"Weather in {city}: Sunny, {temp}°{units.upper()[0]}"

# 4. Create the Agent
# IMPORTANT: We must add 'calc' to this list for the agent to know it exists!
tools = [calculate_area, calc, get_weather]

# We use create_react_agent which returns a CompiledGraph suitable for .invoke({"messages": ...})
agent = create_agent(llm, tools=tools)

def run_demo():
    print("--- Tool Creation Demo with Qwen3:14b ---")
    
    # Query 1: Basic Math
    query1 = "Calculate the area of a room that is 5.5 meters long and 4 meters wide."
    print(f"\nUser: {query1}")
    response1 = agent.invoke({"messages": [HumanMessage(content=query1)]})
    print(f"Agent: {response1['messages'][-1].content}")

    # Query 2: Custom Name Tool (String Counter)
    query2 = "Count the characters in the word 'Mississippi'"
    print(f"\nUser: {query2}")
    response2 = agent.invoke({"messages": [HumanMessage(content=query2)]})
    print(f"Agent: {response2['messages'][-1].content}")

    # Query 3: Pydantic Tool
    query3 = "What is the weather like in Tokyo? Please tell me in fahrenheit."
    print(f"\nUser: {query3}")
    response3 = agent.invoke({"messages": [HumanMessage(content=query3)]})
    print(f"Agent: {response3['messages'][-1].content}")

if __name__ == "__main__":
    try:
        run_demo()
    except Exception as e:
        print(f"Error: {e}")