# llm-agents

## PART 1: Planning
Chain of thought. Model is asked to think step by step, enabling self-correction.

There are four prompt sequences / steps involved in this project's planning phase:
- Find solutions 
- Review them
- Deep thought over solutions
- Rank results


## PART 2: Make use of Memory
## Memory in LLM Prompting & Agents

**Sensory Memory:** This component captures immediate sensory inputs, triggering the model's processing. Ex.- A prompt

**Short-Term Memory:** Holds temporary information, maintaining context and coherence during interactions. Ex.- Recent Chat History

**Long-Term Memory:** Stores factual knowledge and procedural instructions, allowing agents to generate informed and relevant outputs. Ex.- Context from VectorDBs


## PART 3: Link Multiple Tools to Agent/s
- Neural tools:
    - LLMs (specialised, instruction-tuned) - Generation, Analysis tasks
    - ML models (linear regression, etc.) - Prediction Tasks
- Symbolic tools:
    - Algorithmic (Mathematical calculations) - LLMs are not perfect with maths
    - APIs (Web search) - connect to 3rd party apps

**Langchain Tools - https://python.langchain.com/v0.2/docs/integrations/tools/**

**Langchain Agent - (LLM + Tool + Prompt). They have specific prompt formats. They expect specific type & number of tools.**

Some of the types of agents offered by LangChain include:
1. Zero-shot ReAct
2. OpenAI functions
3. Self-ask with Search
4. Conversational
5. Structured Input ReAct


