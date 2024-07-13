from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub

import os
from dotenv import load_dotenv
load_dotenv('llm_init.env')
model = "gemini-1.5-flash"
api_key = os.getenv("GOOGLE_API_KEY")
llm_gemini = ChatGoogleGenerativeAI(google_api_key=api_key, model=model, temperature=0.7)


## CHAIN OF THOUGHTS - PLANNING
cot_step1 = hub.pull("rachnogstyle/nlw_jan24_cot_step1")
cot_step2 = hub.pull("rachnogstyle/nlw_jan24_cot_step2")
cot_step3 = hub.pull("rachnogstyle/nlw_jan24_cot_step3")
cot_step4 = hub.pull("rachnogstyle/nlw_jan24_cot_step4")

# create LLMChains
def create_llm_chain(prompt, output_key):
    return LLMChain(
        llm=llm_gemini,
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


## RESULTS
# > Entering new SequentialChain chain...

# > Finished chain.
# {'input': 'human colonization of Mars', 
# 'perfect_factors': 'The distance between Earth and Mars is very large, making regular resupply difficult', 
# 'ranked_solutions': 
#   "## Ranking of Solutions:\n\nBased on the deeper analysis, here's a ranking of the solutions in order of promise, along with justifications and considerations:
# \n\n**1. In-Situ Resource Utilization (ISRU)**\n\n**Justification:** ISRU offers the most immediate and practical approach to resource independence on Mars. Utilizing Martian resources for water, oxygen, and building materials reduces reliance on Earth-based supplies, significantly lowering transportation costs and environmental impact. 
# \n\n**Considerations:** \n* **Resource Availability:** The success of ISRU hinges on the abundance and accessibility of resources. Extensive exploration and mapping are crucial to identify reliable resource deposits.
# \n* **Technology Development:** While some ISRU technologies are mature, others require further development and testing. Investing in research and development is essential for improving efficiency and reliability.\n* 
# **Environmental Impact:** ISRU must be conducted responsibly to avoid negative impacts on the Martian ecosystem. Careful planning and continuous monitoring are crucial to ensure sustainable resource utilization.
# 
# \n\n**2. Self-Sustaining Bioregenerative Life Support Systems**\n\n**Justification:** Bioregenerative systems offer a path towards long-term sustainability by creating closed-loop ecosystems that recycle waste and produce food. This approach reduces the need for constant resupply from Earth and fosters a more self-sufficient Martian colony.
# \n\n**Considerations:**\n* **System Complexity:** Bioregenerative systems are complex and require careful management and monitoring to maintain stability. Investing in robust control systems and redundancy is crucial.\n* **Disease and Pests:** Potential for disease outbreaks or pest infestations requires robust biosecurity measures and ongoing research into resilient crops and microbial communities.\n* **Food Diversity:
# ** Initial bioregenerative systems may offer limited food variety. Research into new crops and cultivation methods is essential for a diverse and healthy diet.
# 
# \n\n**3. Advanced Propulsion Technologies**\n\n**Justification:** While advanced propulsion technologies are still in their early stages of development, they hold the key to unlocking future possibilities for interstellar travel and expansion. These technologies could dramatically reduce travel time and enable missions to more distant destinations.
# \n\n**Considerations:**\n* **Technological Feasibility:**  Advanced propulsion technologies face significant technical challenges. Continued investment in research and development is crucial for achieving breakthroughs.\n* 
# **Safety Concerns:**  Nuclear fusion and antimatter propulsion pose potential risks. Robust safety protocols and mitigation strategies are essential.\n* 
# **High Development Costs:**  Advanced propulsion technologies require significant investment in research, development, and testing. International collaboration and private sector investment are crucial for funding these endeavors.
# 
# \n\n**Final Thoughts:**\n\nEach solution offers unique advantages and challenges. A combined approach, embracing innovation and collaboration, will be essential for achieving sustainable Martian colonization. ISRU and bioregenerative life support systems provide immediate and achievable goals for resource independence and self-sufficiency, while advanced propulsion technologies pave the way for future exploration and expansion. Continued research, development, and international cooperation are essential for realizing these ambitious goals. \n"}