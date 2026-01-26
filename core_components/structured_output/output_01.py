# ============================================================================
# Structured Output Example with LangChain
# Demonstrates how to extract and structure data using Pydantic models
# and LangChain agents with response formatting
# ============================================================================

from pydantic import BaseModel, Field
from langchain.agents.structured_output import ToolStrategy
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from typing import Literal

# Pydantic model for structured contact information extraction
class ContactInfo(BaseModel):
    """Contact information for a person."""
    name: str = Field(description="The name of the person")
    email: str = Field(description="The email address of the person")
    phone: str = Field(description="The phone number of the person")


# Pydantic model for extracting action items from meeting transcripts
class MeetingAction(BaseModel):
    """Action items extracted from a meeting transcript."""
    task: str = Field(description="The specific task to be completed")
    assignee: str = Field(description="Person responsible for the task")
    priority: Literal["low", "medium", "high"] = Field(description="Priority level")


# Initialize the LLM model (Qwen 3 14B via Ollama)
# temperature=0 ensures deterministic outputs
llm = ChatOllama(model="qwen3:14b", temperature=0)

# ============================================================================
# Example 1: Simple structured output with auto-selected provider strategy
# Uncomment below to extract contact information directly
# ============================================================================
# agent = create_agent(
#     model=llm,
#     response_format=ContactInfo  # Auto-selects ProviderStrategy
# )

# result = agent.invoke({
#     "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
# })

# print(result["structured_response"])
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')

# ============================================================================
# Example 2: Structured output with ToolStrategy for meeting action items
# Uses custom error handling and tool messaging
# ============================================================================
agent = create_agent(
    model=llm,
    tools=[],
    response_format=ToolStrategy(
        schema=MeetingAction,
        handle_errors="Please provide a valid rating between 1-5 and include a comment.", #Error handling example
        tool_message_content="Action item captured and added to meeting notes!"
    )
)

# Invoke the agent with a meeting transcript
result = agent.invoke({
    "messages": [{"role": "user", "content": "From our meeting: Sarah needs to update the project timeline as soon as possible"}]
})

# Print the structured action item extracted from the meeting
print(result)