import os

from typing import Annotated, Sequence, List, Literal # For type hints and better code readability
from langgraph.graph import StateGraph, START, END, MessagesState 
from langgraph.types import Command
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
# from langchain_community.tools.riza.command import ExecPython
from langgraph.prebuilt import create_react_agent

from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# tool_code_interpreter = ExecPython()

llm = ChatGroq(groq_api_key = groq_api_key, model_name = "llama-3.3-70b-versatile")

def code_node(state : MessagesState) -> Command[Literal["validator"]]:
    """
    Coder node for leveraging a ReAct agent to process analyzing, solving math questions and executing code.

    Args: 
        state (MessagewsState) : The current state containing the conversation history.

    Returns:
        Command: A command to update the state with research results and route to the validator.

    """

    # create a specialized ReAct agent for coding and problem - solving tasks
    code_agent = create_react_agent(
        model = llm
        tools = [tool_code_interpreter],
        # state_modifier = (
        #     "You are a coder and analyst. Focus on mathematical calculations, analyzing, solving math questions, " \
        #     "and executing code. Handle technical problem - solving and data tasks"
        # )
    )

    result = code_agent.invoke(state)

    print(f"Current Node : Coder -> Goto : validator")

    return Command(
        update = {
            "messages" : [
                # Append the last message (agent's response) to the state, tagged with "coder"
                HumanMessage(content = result["messages"][-1].content, name = "coder")
            ]
        },

        goto = "validator",
    )
