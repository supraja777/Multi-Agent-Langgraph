import os

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END, MessagesState 
from langgraph.types import Command
from langchain_groq import ChatGroq

from dotenv import load_dotenv
from typing import Annotated, Sequence, List, Literal # For type hints and better code readability
from pydantic import BaseModel, Field # 'BaseModel' is the base class used to create data models, 'Field' is used to provide additional metadata

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key = groq_api_key, model_name = "llama-3.3-70b-versatile")

# System prompt providing clear instructions to the validator agent
system_prompt = '''
You are a workflow validator. Your task is to ensure the quality of the workflow. Specifically, you must:
- Review the user's question (the first message in the workflow).
- Review the answer (the last message in the workflow).
- If the answer satisfactorily addresses the question, signal to end the workflow.
- If the answer is inappropriate or incomplete, signal to route back to the supervisor for re-evaluation or further refinement.
Ensure that the question and answer match logically and the workflow can be concluded or continued based on this evaluation.

Routing Guidelines:
1. 'supervisor' Agent: For unclear or vague state messages.
2. Respond with 'FINISH' to end the workflow.
'''

class Validator(BaseModel):
    next: Literal["supervisor", "FINISH"] = Field(
        description = "Specifies the next worker in the pipeline : 'supervisor' to continue or 'FINISH' to terminate"
    )

    reason: str = Field(
        description = "The reason for the decision."
    )

def validator_node(state : MessagesState) -> Command[Literal["supervisor", "__end__"]]:
    """
        Validator node for checking if the question and  the answer are appropriate.

        Args:
            state (MessagesState) : The current state containing message history
        Returns:
            Command: A command indicating whether to route back to the supervisor or end the workflow
 
    """

    # Extract the first (user's question) and the last (agent's response) messages
    user_question = state["messages"][0].content
    agent_answer = state["messages"][-1].content

    # Prepare the message history with system prompt

    messages = [
        {"role" : "system", "content" : system_prompt},
        {"role": "user", "content": user_question},
        {"role" : "assistant", "content" : agent_answer},
    ]

    # Invoke the LLM with structured output using the Validator schema
    response = llm.with_structured_output(Validator).invoke(messages)

    # Extract the 'next' routing decision and the 'reason' from the response
    goto = response.next
    reason = response.reason

    if goto == "FINISH" or goto == "END":
        goto = END # Transition to the termination state
        print("Transitioning to END") 
    else:
        print(f"Current Node: Validator -> Goto: Supervisor")
    
    return Command (
        update = {
            "messages" : [
                HumanMessage(content = reason, name = "validator")
            ]
        },

        goto = goto,
    )
