import os

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END, MessagesState
from typing import Annotated, Sequence, List, Literal
from langgraph.types import Command
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent # Prebuilt tools and agents for streamlined dev
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

tool_tavily = TavilySearchResults(max_results=2)

llm = ChatGroq(groq_api_key = groq_api_key, model_name = "llama-3.3-70b-versatile")

def research_node(state : MessagesState) -> Command[Literal["validator"]]:
    """
    Research node for leveraging a ReAct agent to process research related tasks.

    Args:
        state (MessagesState): The current state containing conversation history

    Returns:
        Command: A command to update the state with research results and route to the validator.

    """

    # # Create a ReAct agent specialized for research tasks
    # research_agent = create_react_agent(
    #     llm,
    #     [tool_tavily], 
    #     "You are a researcher. Focus on gathering information and generating content. Do not perform any other tasks" 
    # )

    research_agent = create_react_agent(
        model=llm,
        tools=[tool_tavily],
        # messages_modifier="You are a researcher. Focus on gathering accurate information and generating concise reports."
    )


    result = research_agent.invoke(state)
    print(f"Current Node: Researcher -> Goto: Validator")

    # Extract the last message from the result and update the state
    return Command(
        update = {
            "messages" : [
                HumanMessage(
                    content = result["messages"][-1].content, # Content of agent's response
                    name = "researcher" # Name of node generating this message
                )
            ]
        },
        goto = "validator", # Route back to validator for further processing
    )

