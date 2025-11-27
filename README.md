
# Groq-Powered Multi-Agent System

A multi-agent LangGraph workflow powered by Groq Llama 3.3 for enhanced reasoning, research, and code execution.

---

## 🚀 Overview

The **Groq-Powered Multi-Agent System** leverages **LangGraph** and **Groq Llama 3.3** to orchestrate a multi-agent workflow capable of handling complex tasks involving:

- Natural language understanding and prompt refinement
- Research and data gathering
- Coding, calculations, and problem-solving
- Validation and workflow supervision

This system is designed for **scalable autonomous reasoning**, where agents collaborate under the guidance of a central supervisor to efficiently complete tasks.

---

## 🤖 About the Multi-Agent System

The Groq-Powered Multi-Agent System is designed as a collaborative network of intelligent agents, each specialized for a distinct task, working together to achieve complex goals. Unlike single-agent workflows, this system leverages the strengths of multiple LLM-powered agents:

This modular, node-based architecture allows agents to communicate through a shared state, enabling dynamic task allocation, error correction, and optimized reasoning.

By distributing responsibilities across specialized agents, the system achieves:

*  Scalable problem-solving – Complex tasks are decomposed into smaller, manageable sub-tasks.

*  Improved accuracy – Each agent focuses on its area of expertise.

*  Robust workflow management – Validation ensures outputs are reliable before completion.

*  Flexibility – New agents or tools can be added seamlessly.

In essence, this multi-agent design mirrors a human team workflow, where collaboration, specialization, and supervision lead to higher efficiency and smarter outputs.

---

## 🏗 Architecture

The system consists of **five main nodes** in a **LangGraph workflow**:

```mermaid
graph TD
    START --> Supervisor
    Supervisor --> Enhancer
    Supervisor --> Researcher
    Supervisor --> Coder
    Enhancer --> Supervisor
    Researcher --> Validator
    Coder --> Validator
    Validator -->|Finish| END
    Validator -->|Re-evaluate| Supervisor
````

* **Supervisor**: Directs tasks to the most suitable agent.
* **Enhancer**: Refines and clarifies user inputs.
* **Researcher**: Gathers accurate information using Groq-powered ReAct agents.
* **Coder**: Executes code, calculations, and technical problem-solving.
* **Validator**: Checks output quality and decides whether to finish or loop back.

---

## ✨ Features

* Multi-agent coordination with **LangGraph**
* **Groq Llama 3.3** integration for powerful reasoning
* ReAct agents for coding, research, and problem-solving
* Workflow supervision and validation for reliable outputs
* Streamlined state management with `MessagesState`
* Modular node-based architecture for extensibility

---

## 📦 Installation

1. **Clone the repository**

```bash
git clone <repo_url>
cd <repo_folder>
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set environment variables**

Create a `.env` file:

```env
GROQ_API_KEY=<your_groq_api_key>
```

---

## 🗂 Folder Structure

```
├── app.py                  # Main entry point for workflow execution
├── supervisor_node.py      # Supervisor node logic
├── enhancer_node.py        # Prompt enhancement node
├── research_node.py        # Research and information gathering node
├── code_node.py            # Coder node with ReAct agent
├── validator_node.py       # Workflow validation node
├── supervisor_node.py
├── .env                    # API keys and configuration
├── requirements.txt        # Python dependencies
```

---

## 🧠 Workflow Explanation

1. **Start** → **Supervisor**

   * Supervisor evaluates the task and routes it to the best-suited agent.

2. **Enhancer Node**

   * Refines user queries for clarity and precision.

3. **Research Node**

   * Uses **TavilySearchResults** + ReAct agent to gather relevant information.

4. **Coder Node**

   * Executes technical tasks, calculations, or code using a ReAct agent.

5. **Validator Node**

   * Checks task output; decides to finish or route back to **Supervisor** for further refinement.

---

## ⚡ Example Usage

```python
from app import graph
import pprint

inputs = {
    "messages": [
        ("user", "Research the impact of climate change on agriculture in Southeast Asia. Propose potential solutions."),
    ]
}

for output in graph.stream(inputs):
    for key, value in output.items():
        if value is None:
            continue
        pprint.pprint(f"Output from node '{key}':")
        pprint.pprint(value, indent=2, width=80, depth=None)
        print()
```

---

## 📝 Node Summary

| Node       | Role                                                          |
| ---------- | ------------------------------------------------------------- |
| Supervisor | Routes tasks to agents based on workflow state                |
| Enhancer   | Refines and clarifies user input                              |
| Researcher | Performs information gathering using Groq-powered ReAct agent |
| Coder      | Executes code, calculations, and problem-solving tasks        |
| Validator  | Validates answers, decides whether to finish or loop back     |

---

## 💡 Advantages

* Structured, multi-agent workflow for **complex tasks**
* High **accuracy and reliability** via validation and supervision
* Modular design allows **adding new nodes or tools** easily
* Leverages **state-of-the-art LLMs** for reasoning, coding, and research

