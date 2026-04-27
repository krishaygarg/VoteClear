import os
import sys
import json,time
import pickle
import re
from langchain_core.rate_limiters import InMemoryRateLimiter
from aiolimiter import AsyncLimiter
from tools import search_tool, save_tool, wiki_tool
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from typing import List, Dict, Any, Union
from langchain_core.messages import BaseMessage
from langchain_tavily import tavily_search
def extract_json(raw_str: str) -> dict:
    cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_str.strip(), flags=re.IGNORECASE)
    return json.loads(cleaned)
def format(raw_str: str) -> str:
    cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_str.strip(), flags=re.IGNORECASE)
    data = json.loads(cleaned)
    output = []
    for response in data.get("responses", []):
        topic = response.get("topic", "Unknown Topic")
        summary = response.get("summary", "No summary available.")
        sources = response.get("sources", [])
        source_text = "\n".join(f"- {src}" for src in sources) if sources else "No sources provided."
        formatted = f"""Topic: {topic}
Summary: {summary}
Sources:
{source_text}"""
        output.append(formatted.strip())
    return "\n\n" + ("\n" + "-" * 40 + "\n\n").join(output)
GEMINI_RPM_LIMIT = 9
LLM_RATE_LIMITER_INSTANCE = InMemoryRateLimiter(
    requests_per_second=GEMINI_RPM_LIMIT / 60, # Convert RPM to RPS
    check_every_n_seconds=0.1, # How often to check for available tokens
    max_bucket_size=GEMINI_RPM_LIMIT # Allow a burst up to the RPM limit
)
class PolicyStance(BaseModel):
    summary: str = Field(description="A comprehensive summary of the candidate's stance.")
    sources: List[str] = Field(description="A list of full URLs to the verifiable sources.")
ELECTION_NAME = "2025 Virginia gubernatorial election"
# # Model for a single candidate's research
# class CandidateResearch(BaseModel):
#     candidate_name: str = Field(description="The full name of the candidate.")
#     stances: List[PolicyStance] = Field(description="A list of policy stances for this candidate.")

# # Model for the collection of all research (for final output)
# class ResearchResponseCollection(BaseModel):
#     candidates_research: List[CandidateResearch] = Field(description="A list of research findings for all candidates.")

load_dotenv()
model = "gemini-2.5-flash"

class CandidateList(BaseModel):
    candidates: List[str] = Field(description="A list of confirmed candidate names for the election.")

candidate_parser = PydanticOutputParser(pydantic_object=CandidateList)
policy_parser = PydanticOutputParser(pydantic_object=PolicyStance)
candidate_identification_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert political researcher. Your sole task is to identify and list all *confirmed* candidates for the {ELECTION_NAME}.

Use the search tool to find this information. Prioritize official election board websites, reputable news organizations, and non-partisan political research sites.

Avoid speculation or unconfirmed rumors. Only list individuals who have officially declared their candidacy or are widely recognized as confirmed candidates by reliable sources.
Your FINAL ANSWER MUST BE STRICTLY a JSON object formatted according to this schema:
{format_instructions}
            """
        ),
        ("placeholder", "{chat_history}"),
        ("human", "Identify all confirmed candidates for the " + ELECTION_NAME),
        ("placeholder","{agent_scratchpad}")
    ]
).partial(format_instructions=candidate_parser.get_format_instructions(), ELECTION_NAME=ELECTION_NAME)
print(candidate_parser.get_format_instructions())
candidate_identification_agent = create_tool_calling_agent(
    llm=ChatGoogleGenerativeAI(model=model, temperature=0, rate_limiter=LLM_RATE_LIMITER_INSTANCE), # Lower temperature for factual retrieval
    prompt=candidate_identification_prompt,
    tools=[wiki_tool] # Only need search tool for this agent
)
candidate_identification_executor = AgentExecutor(
    agent=candidate_identification_agent,
    tools=[wiki_tool],
    verbose=True
)


PREDEFINED_POLICY_AREAS = [
    "Economic Policy: This category includes government actions aimed at influencing the economy's performance and stability. It covers areas like taxation, government spending, trade regulations, monetary policy, jobs, and inflation.",
    "Social Policy: These policies focus on the well-being and welfare of the population. This broad category includes areas like healthcare, education, social security programs, housing, and policies aimed at addressing poverty and inequality.",     
"Environmental & Energy Policy: This category deals with the protection of the environment, the management of natural resources, and the development and regulation of energy sources. This includes policies related to climate change, conservation, pollution control, and the transition to renewable energy.",
"Foreign Affairs & National Security: These policies govern a nation's interactions with other countries and its defense. This includes diplomacy, immigration policy, defense spending, intelligence activities, international trade agreements, and responses to global threats."
]

policy_research_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert political researcher focused on providing objective, factual information about the stances of a given candidate on specific policy issues.

Your task is to research the current positions of "{candidate_name}" on ONLY the following policy area:
{policy_area}

Provide detailed, comprehensive information and cite the specific source(s) (including full URLs) where the information was found. If a stance has evolved or is nuanced, reflect that accurately.
Invoke the search tool to gather information.

Use the following output format: \n
{format_instructions}
            """
        ),
        ("placeholder", "{chat_history}"),
        ("human", "Research the policy stances for {candidate_name} on this policy area:\n {policy_area}"),
        ("placeholder","{agent_scratchpad}")
    ]
).partial(format_instructions=policy_parser.get_format_instructions())

policy_research_agent = create_tool_calling_agent(
    llm=ChatGoogleGenerativeAI(model=model, temperature=0.2, rate_limiter=LLM_RATE_LIMITER_INSTANCE), # Allow some creativity for summarization
    prompt=policy_research_prompt,
    tools=[wiki_tool, search_tool] # Save tool will be used in the orchestration logic, not directly by this agent
)
policy_research_executor = AgentExecutor(
    agent=policy_research_agent,
    tools=[wiki_tool, search_tool],
    verbose=True
)

async def run_mayoral_research_workflow():
    # print("Step 1: Identifying confirmed candidates...")
    # try:
    #     candidate_identification_response = await candidate_identification_executor.ainvoke({"query": "Identify all confirmed candidates for the " + ELECTION_NAME})
    #     candidate_list_obj = candidate_parser.parse(candidate_identification_response["output"])
    #     confirmed_candidates = candidate_list_obj.candidates
    #     print(f"Identified Candidates: {confirmed_candidates}")
    # except Exception as e:
    #     print(f"Error identifying candidates: {e}")
    #     return

    # if not confirmed_candidates:
    #     print("No confirmed candidates found. Exiting.")
    #     return
    # sys.exit()
    confirmed_candidates = ["Winsome Earle-Sears", "Abigail Spanberger"]#, "Eric Adams", "Andrew Cuomo"]
    all_candidates_research_data = []

    print("\nStep 2: Researching policy stances for each candidate...")
    for candidate_name in confirmed_candidates:
        print(f"\n--- Researching: {candidate_name} ---")
        candidate_data = []
        for policy_area in PREDEFINED_POLICY_AREAS:
                policy_research_response = await policy_research_executor.ainvoke({
                    "candidate_name": candidate_name,
                    "policy_area": policy_area,
                    "query": f"Research the policy stances for {candidate_name} on the following policy area: \n {policy_area}."
                })
                candidate_research_obj = PydanticOutputParser(pydantic_object=PolicyStance).parse(policy_research_response["output"])
                candidate_data.append(candidate_research_obj)
        all_candidates_research_data.append(candidate_data)
    print("\nStep 3: Compiling results into text file...")
    final_output_content = "Candidate Stances**\n\n"

    for i in range(len(all_candidates_research_data)):
        final_output_content += "---\n\n"
        final_output_content += f"**Candidate Name: {confirmed_candidates[i]}**\n\n"
        for j in range(len(PREDEFINED_POLICY_AREAS)):
            final_output_content += f"**{PREDEFINED_POLICY_AREAS[j]}:**\n"
            final_output_content += f"{all_candidates_research_data[i][j].summary}\n"
            final_output_content += f"Source: {', '.join(all_candidates_research_data[i][j].sources)}\n\n"

    # Use your save_tool to write the content to a file
    print(final_output_content)
    with open('output1.txt', 'w') as f: f.write(final_output_content)
    with open('research.pkl', 'wb') as f: pickle.dump(all_candidates_research_data, f)
    with open('candidates.pkl', 'wb') as f: pickle.dump(confirmed_candidates, f)
    with open('policyareas.pkl', 'wb') as f: pickle.dump(PREDEFINED_POLICY_AREAS, f)


import asyncio
asyncio.run(run_mayoral_research_workflow())


# parser = PydanticOutputParser(pydantic_object = ResearchResponseCollection)
# llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")
# prompt = ChatPromptTemplate.from_messages(
#     [
#     (
#         "system",
#         """
# You are an expert political researcher focused on providing objective, factual information about the stances of each candidate in the upcoming 2025 New York City Mayoral election.

# Your primary goal is to:

# Identify all confirmed candidates for the 2025 New York City Mayoral election. Format the output like this: \n {format_instructions}
# For each identified candidate, research and summarize their current positions on key policy issues. The agent should autonomously determine which policy areas are most relevant and significant for each individual candidate, based on available public information (e.g., campaign websites, public statements, legislative history).

# After completing research for all candidates and their relevant policy areas, compile all findings into a single, comprehensive text file.

# Information Gathering Guidelines:

# Invoke the search tool to gather information.

# Prioritize information from:

# Official government websites (e.g., NYC.gov, candidate's official campaign sites).

# Reputable, non-partisan research organizations (e.g., Citizens Budget Commission, Regional Plan Association).

# Major news organizations known for their factual reporting and journalistic integrity (e.g., Associated Press, Reuters, The New York Times, Wall Street Journal).

# Voting records and legislative actions (if applicable to a candidate's past roles).

# Avoid: Opinion pieces, editorials, Wikipedia, highly partisan news sources, speculation, personal interpretations, or information that cannot be directly attributed to a verifiable source.

# Output Format for Each Policy Stance:
# For each policy stance identified, provide a 3-4 sentence summary and cite the specific source(s) (including full URLs) where the information was found. If a stance has evolved or is nuanced, reflect that accurately.

# Final Output:
# Save all collected research into a single text file. Format the output like this: \n {format_instructions}
#        """
#     ),
#     ("placeholder", "{chat_history}"),
#     ("human", "{query}"),
#     ("placeholder","{agent_scratchpad}"),
#     ]
# ).partial(format_instructions=parser.get_format_instructions())
# tools = [wiki_tool, save_tool]
# agent = create_tool_calling_agent(
#     llm=llm,
#     prompt=prompt,
#     tools=tools
# )
# agent_executor = AgentExecutor(agent=agent, tools=tools,verbose=True)
# query= "Follow instructions in the prompt to research candidates in the NYC Mayoral Election 2025."
# raw_response=agent_executor.invoke({"query": query})
# print(raw_response)
# print(json.dumps(extract_json(raw_response["output"])))
# json_data = json.loads(re.search(r'```json\n(.*?)\n```', raw_response["output"], re.DOTALL).group(1))
# with open("output.json", "w") as f:
#     json.dump(json_data, f, indent=4)

