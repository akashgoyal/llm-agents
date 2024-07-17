# llm-agents


## **1: intro_to_llm_agents**

It is structured in three parts :
1. Planning / Reasoning
2. Make use of Memory - Long, Short, Sensory
3. Link Multiple Tools to Agent/s

---

## **2: base_code_llm_agents_langchain**

To build llm agents using langchain, refer this repository for base code. It covers :
1. Popular LLM initialization
2. Search tools initialization
3. Agent with multiple tools using different LLMs.

This makes it easy to understand the flow and extend codebase.

---


## **4: text2pandas-query-pl**

Talk to your Dataset. This project uses 'LlamaIndex Query Pipelines'.
1. Initialize LLM. Read dataset file.
2. Define Query Pipeline, which is actually a DAG flow. Take care of output links. Visualise the DAG.
3. Talk to your llm app.

`input text -> pandas commands -> eval -> result`

---

