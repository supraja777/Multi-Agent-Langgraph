import os

from typing import Annotated, Sequence, List, Literal # For type hints and better code readability

from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.riza.command import ExecPython
from pydantic import BaseModel, Field # 'BaseModel' is the base class used to create data models, 'Field' is used to provide additional metadata

from langchain_core.messages import HumanMessage

from langgraph.graph import StateGraph, START, END, MessagesState 
from langgraph.types import Command

from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key = groq_api_key, model_name = "llama-3.3-70b-versatile")

system_prompt = ('''You are a workflow supervisor managing a team of three agents: Prompt Enhancer, Researcher, and Coder. 
                 
                Your role is to direct the flow of tasks by selecting the next agent based on the current stage of the workflow. 
                 
                For each task, provide a clear rationale for your choice, ensuring that the workflow progresses logically, efficiently, and toward a timely completion.

**Team Members**:
1. Enhancer: Use prompt enhancer as the first preference, to Focus on clarifying vague or incomplete user queries, improving their quality, 
              and ensuring they are well-defined before further processing.
2. Researcher: Specializes in gathering information.
3. Coder: Handles technical tasks related to caluclation, coding, data analysis, and problem-solving, ensuring the correct implementation of solutions.

**Responsibilities**:
1. Carefully review each user request and evaluate agent responses for relevance and completeness.
2. Continuously route tasks to the next best-suited agent if needed.
3. Ensure the workflow progresses efficiently, without terminating until the task is fully resolved.

Your goal is to maximize accuracy and effectiveness by leveraging each agent’s unique expertise while ensuring smooth workflow execution.
''')

class Supervisor(BaseModel):

    next: Literal["enhancer", "researcher", "coder"] = Field(
        description = "Specifies the next worker in the pipeline: " \
        "'enhancer' for enhancing the user prompt if it is unclear or vague, " \
        "'researcher' for additional information gathering, " \
        "'coder' for solving technical or code-related problems"
    )
    reason: str = Field(
        description = "The reason for the decision, providing context on why a particular worker was chosen."
    )

# Define the supervisor node function to handle state transitions
def supervisor_node(state : MessagesState) -> Command[Literal["enhancer", "researcher", "coder"]]:
    """Supervisor node for routing tasks based on current state and LLM response.

    Args:
        state (MessagesState) : The current state containing messsage history.
    Returns:
        Command: A command indicating the next state or action.
    
    """

    # Prepare messsages by appending the system prompt to the message history
    messages = [
        {"role" : "system", "content" : system_prompt}, # System-Level instructions or context and appending previous messages from the state

    ] + state["messages"]

    response = llm.with_structured_output(Supervisor).invoke(messages)

    goto = response.next
    reason = response.reason

    print(f"Current Node: Supervisor -> Goto: {goto}")

    return Command(
        update = {
            "messages" : [
                # Append the reason (supervisor's response) to the state, tagged with "supervisor"
                HumanMessage(content = reason, name = "supervisor")
            ]
        },

        goto = goto, # specify the next node in the workflow
    )
