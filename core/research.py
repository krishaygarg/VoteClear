import asyncio
import json
import re
import os
from typing import List

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# Relative imports inside the core package
from .tools import search_tool, wiki_tool
from . import db

load_dotenv()

MODEL_NAME = "gemini-2.5-flash"
GEMINI_RPM_LIMIT = 4

llm_rate_limiter = InMemoryRateLimiter(
    requests_per_second=GEMINI_RPM_LIMIT / 60,
    check_every_n_seconds=0.1,
    max_bucket_size=GEMINI_RPM_LIMIT
)

class PolicyStance(BaseModel):
    summary: str = Field(description="A comprehensive summary of the candidate's stance.")
    sources: List[str] = Field(description="A list of full URLs to the verifiable sources.")

class CandidateList(BaseModel):
    candidates: List[str] = Field(description="A list of confirmed candidate names for the election.")

candidate_parser = PydanticOutputParser(pydantic_object=CandidateList)
policy_parser = PydanticOutputParser(pydantic_object=PolicyStance)

PREDEFINED_POLICY_AREAS = [
    "Economic Policy: This category includes government actions aimed at influencing the economy's performance and stability. It covers areas like taxation, government spending, trade regulations, monetary policy, jobs, and inflation.",
    "Social Policy: These policies focus on the well-being and welfare of the population. This broad category includes areas like healthcare, education, social security programs, housing, and policies aimed at addressing poverty and inequality.",
    "Environmental & Energy Policy: This category deals with the protection of the environment, the management of natural resources, and the development and regulation of energy sources. This includes policies related to climate change, conservation, pollution control, and the transition to renewable energy.",
    "Foreign Affairs & National Security: These policies govern a nation's interactions with other countries and its defense. This includes diplomacy, immigration policy, defense spending, intelligence activities, international trade agreements, and responses to global threats."
]

# Set up Candidate Identification Agent
candidate_identification_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert political researcher. Your sole task is to identify and list all *confirmed* major candidates for the {election_name}.

Use the search tool or Wikipedia to find this information. Prioritize official election board websites, reputable news organizations, and non-partisan political research sites.

Avoid speculation or unconfirmed rumors. Only list individuals who have officially declared their candidacy or are widely recognized as confirmed major candidates by reliable sources.
Your FINAL ANSWER MUST BE STRICTLY a JSON object formatted according to this schema:
{format_instructions}"""
    ),
    ("placeholder", "{chat_history}"),
    ("human", "Identify all confirmed candidates for the {election_name}"),
    ("placeholder", "{agent_scratchpad}")
])

candidate_identification_agent = create_tool_calling_agent(
    llm=ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0, rate_limiter=llm_rate_limiter),
    prompt=candidate_identification_prompt,
    tools=[wiki_tool, search_tool]
)

candidate_identification_executor = AgentExecutor(
    agent=candidate_identification_agent,
    tools=[wiki_tool, search_tool],
    verbose=True
)

# Set up Policy Research Agent
policy_research_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert political researcher focused on providing objective, factual information about the stances of a given candidate on specific policy issues.

Your task is to research the current positions of "{candidate_name}" on ONLY the following policy area:
{policy_area}

Provide detailed, comprehensive information and cite the specific source(s) (including full URLs) where the information was found. If a stance has evolved or is nuanced, reflect that accurately.
Invoke the search tool to gather information.

Use the following output format: \n
{format_instructions}"""
    ),
    ("placeholder", "{chat_history}"),
    ("human", "Research the policy stances for {candidate_name} on this policy area:\n {policy_area}"),
    ("placeholder", "{agent_scratchpad}")
]).partial(format_instructions=policy_parser.get_format_instructions())

policy_research_agent = create_tool_calling_agent(
    llm=ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.2, rate_limiter=llm_rate_limiter),
    prompt=policy_research_prompt,
    tools=[wiki_tool, search_tool]
)

policy_research_executor = AgentExecutor(
    agent=policy_research_agent,
    tools=[wiki_tool, search_tool],
    verbose=True
)

async def identify_candidates(election_name: str) -> List[str]:
    """Identifies major confirmed candidates for an election using the agent."""
    print(f"Identifying candidates for: {election_name}...")
    try:
        response = await candidate_identification_executor.ainvoke({
            "election_name": election_name,
            "chat_history": [],
            "format_instructions": candidate_parser.get_format_instructions()
        })
        parsed = candidate_parser.parse(response["output"])
        print(f"Identified candidates: {parsed.candidates}")
        return parsed.candidates
    except Exception as e:
        print(f"Error identifying candidates: {e}")
        return []

async def research_candidate_and_save(election_id: str, candidate_name: str) -> int:
    """Researches the stances of a candidate and saves them to SQLite."""
    print(f"\n--- Researching: {candidate_name} for election {election_id} ---")
    
    candidate_id = db.save_candidate(election_id, candidate_name)
    
    for policy_area in PREDEFINED_POLICY_AREAS:
        print(f"  Researching policy: {policy_area[:30]}...")
        try:
            policy_research_response = await policy_research_executor.ainvoke({
                "candidate_name": candidate_name,
                "policy_area": policy_area,
                "chat_history": [],
                "query": f"Research the policy stances for {candidate_name} on the following policy area: \n {policy_area}."
            })
            candidate_research_obj = policy_parser.parse(policy_research_response["output"])
            
            db.save_policy_stance(
                candidate_id=candidate_id,
                policy_area=policy_area,
                summary=candidate_research_obj.summary,
                sources=candidate_research_obj.sources
            )
        except Exception as e:
            print(f"  Error researching policy area for {candidate_name}: {e}")
            db.save_policy_stance(
                candidate_id=candidate_id,
                policy_area=policy_area,
                summary="Information currently unavailable.",
                sources=[]
            )
            
    return candidate_id

async def run_research_workflow():
    db.init_db()
    election_id = "2025-virginia-gubernatorial-election"
    election_name = "2025 Virginia Gubernatorial Election"
    db.save_election(election_id, election_name, "2025-11-04")
    
    candidates = await identify_candidates(election_name)
    if not candidates:
        candidates = ["Winsome Earle-Sears", "Abigail Spanberger"]
        
    for candidate in candidates:
        await research_candidate_and_save(election_id, candidate)
        
    print("\nResearch complete and saved to database!")
    
    # Write compatibility text file (saved in root parent folder/output)
    output_content = db.get_all_research_for_election(election_id)
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "output1.txt"), "w", encoding="utf-8") as f:
        f.write(output_content)

if __name__ == "__main__":
    asyncio.run(run_research_workflow())
