from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chains import SequentialChain

import json

## CHAIN OF THOUGHTS - PLANNING
# read prompts json
planning_prompts_json_path = "prompts/planning_prompts.json"
with open(planning_prompts_json_path, 'r') as f:
    planning_prompts = json.load(f)

cot_step1 = planning_prompts["cot1"]
cot_step2 = planning_prompts["cot2"]
cot_step3 = planning_prompts["cot3"]
cot_step4 = planning_prompts["cot4"]


# create LLMChains
model = "gpt-3.5-turbo"
def create_llm_chain(prompt, output_key):
    return LLMChain(
        llm=ChatOpenAI(temperature=0, model=model),
        prompt=prompt,
        output_key=output_key
    )

# Create chains
chain1 = create_llm_chain(prompt=cot_step1, output_key="solutions")
chain2 = create_llm_chain(prompt=cot_step2, output_key="review")
chain3 = create_llm_chain(prompt=cot_step3, output_key="deepen_thought_process")
chain4 = create_llm_chain(prompt=cot_step4, output_key="ranked_solutions")


# run sequential chains
overall_chain = SequentialChain(
    chains=[chain1, chain2, chain3, chain4],
    input_variables=["input", "perfect_factors"],
    output_variables=["ranked_solutions"],
    verbose=True
)

print(
    overall_chain(
        {
            "input": "human colonization of Mars",
            "perfect_factors": "The distance between Earth and Mars is very large, making regular resupply difficult"
        }
    )
)

## CHAIN OF THOUGHTS - PLANNING - FINISH 


# # prompt review
# from langchain import hub
# # ReAct prompt 
# prompt = hub.pull("hwchase17/react")
# # self-ask with search prompt 
# prompt = hub.pull("hwchase17/self-ask-with-search")
