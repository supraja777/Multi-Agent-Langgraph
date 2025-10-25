import os
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END, MessagesState 
from langgraph.types import Command
from typing import Annotated, Sequence, List, Literal # For type hints and better code readability

from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key = groq_api_key, model_name = "llama-3.3-70b-versatile")

def enhancer_node(state : MessagesState) -> Command[Literal["supervisor"]]:
    """
        Enhancer node for refining and clarifying user inputs.

        Args:
            state (MessagesState) : The current state containing the conversation history
        
        Returns:
            Command : A command to update the state with the enhanced query and route back to supervisor.
        
    """

    # Define the system prompt to guide the LLM in query enhancement
    system_prompt = (
        "You are an advanced query enhancer. Your task is to  "
        ": Select the most appropriate prompt "
        "1 Clarify and refine user inputs " \
        "2 Identify any ambiguities in the query " \
        "3. Generate a more precise and actionable version of the original request"
    )

    # combine the system prompt with the current conversation messages
    messages = [
        {"role" : "system", "content" : system_prompt}, # Provide the context for LLM

    ] + state["messages"] # Include the conversation history for context

    # Use the LLM to process the messages and generate an enhanced query

    enhanced_query = llm.invoke(messages)
    print(f"Current Node : Prompt Enhancer -> Goto: Supervisor")

    # Return a command to update the state with the enhanced query and route back to the supervisor
    
    return Command(
        update={
            "messages":[ # append the enhanced query to the message history
                HumanMessage(
                    content = enhanced_query.content, # content of enhanced query
                    name = "enhancer" # Name of the node processing the message
                )
            ]
        }, 

        goto = "supervisor", # Route to the supervisor for further processing
    )


    
