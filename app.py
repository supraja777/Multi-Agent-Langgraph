import os

from typing import Annotated, Sequence, List, Literal # For type hints and better code readability

from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field # 'BaseModel' is the base class used to create data models, 'Field' is used to provide additional metadata

from langchain_core.messages import HumanMessage

from langgraph.graph import StateGraph, START, END, MessagesState 
from langgraph.types import Command
from IPython.display import Image, display # Import for displaying images 

from dotenv import load_dotenv

from supervisor_node import *
from enhancer_node import *
from research_node import *
from code_node import *
from validator_node import *

import pprint

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


llm = ChatGroq(groq_api_key = groq_api_key, model_name = "llama-3.3-70b-versatile")

tool_tavily = TavilySearchResults(max_results=2)

tool_code_interpreter = ExecPython()

tools = [tool_tavily, tool_code_interpreter]

builder = StateGraph(MessagesState)

builder.add_node("supervisor", supervisor_node) # add supervisor node to the graph

# task specific nodes for various roles
builder.add_node("enhancer", enhancer_node) # Node for refining and clarifying user inputs
builder.add_node("researcher", research_node) # Node for handling research related tasks
builder.add_node("coder", code_node) # Node for managing coding and analytical tasks
builder.add_node("validator", validator_node) # Node for managing coding and analytical tasks

builder.add_edge(START, "supervisor") # Connect the start node to the supervisor node

# Compile the graph to finalize its structure
graph = builder.compile()

# display(Image(graph.get_graph(xray = True).draw_mermaid_png())) # Display the graph's PNG rep
# mermaid_code = graph.get_graph(xray=True).draw_mermaid()
# print(mermaid_code)


# inputs = {
#     "messages" : [
#         ("user", "Weather in Chicago?"),
#     ]
# }

# for output in graph.stream(inputs):
#     for key, value in output.items():
#         if value is None:
#             continue

#         pprint.pprint(f"Output from node '{key} : ")
#         pprint.pprint(value, indent = 2, width = 80, depth = None)
#         print()



# inputs = {
#     "messages": [
#         ("user", "Research the impact of climate change on agriculture in Southeast Asia. Based on your findings, propose potential solutions to mitigate its effects on crop production"),
#     ]
# }
# for output in graph.stream(inputs):
#     for key, value in output.items():
#         if value is None:
#             continue
#         pprint.pprint(f"Output from node '{key}':")
#         pprint.pprint(value, indent=2, width=80, depth=None)
#         print()




inputs = {
    "messages": [
        ("user", "give me how many A's present in a string of AVYGABAAHKJHDAAAAUHBU  ?"),
    ]
}
for output in graph.stream(inputs):
    for key, value in output.items():
        if value is None:
            continue
        pprint.pprint(f"Output from node '{key}':")
        pprint.pprint(value, indent=2, width=80, depth=None)
        print()
     