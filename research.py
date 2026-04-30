import asyncio
import json
import pickle
import re
from typing import List

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from tools import search_tool, wiki_tool
def format_research_output(raw_str: str) -> str:
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
llm_rate_limiter = InMemoryRateLimiter(
    requests_per_second=GEMINI_RPM_LIMIT / 60,
    check_every_n_seconds=0.1,
    max_bucket_size=GEMINI_RPM_LIMIT
)
load_dotenv()

ELECTION_NAME = "2025 Virginia gubernatorial election"
MODEL_NAME = "gemini-2.5-flash"


class PolicyStance(BaseModel):
    summary: str = Field(description="A comprehensive summary of the candidate's stance.")
    sources: List[str] = Field(description="A list of full URLs to the verifiable sources.")
class CandidateList(BaseModel):
    candidates: List[str] = Field(description="A list of confirmed candidate names for the election.")

candidate_parser = PydanticOutputParser(pydantic_object=CandidateList)
policy_parser = PydanticOutputParser(pydantic_object=PolicyStance)

candidate_identification_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert political researcher. Your sole task is to identify and list all *confirmed* candidates for the {ELECTION_NAME}.

Use the search tool to find this information. Prioritize official election board websites, reputable news organizations, and non-partisan political research sites.

Avoid speculation or unconfirmed rumors. Only list individuals who have officially declared their candidacy or are widely recognized as confirmed candidates by reliable sources.
Your FINAL ANSWER MUST BE STRICTLY a JSON object formatted according to this schema:
{format_instructions}"""
    ),
    ("placeholder", "{chat_history}"),
    ("human", "Identify all confirmed candidates for the " + ELECTION_NAME),
    ("placeholder", "{agent_scratchpad}")
]).partial(format_instructions=candidate_parser.get_format_instructions(), ELECTION_NAME=ELECTION_NAME)

candidate_identification_agent = create_tool_calling_agent(
    llm=ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0, rate_limiter=llm_rate_limiter),
    prompt=candidate_identification_prompt,
    tools=[wiki_tool]
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

async def run_research_workflow():
    confirmed_candidates = ["Winsome Earle-Sears", "Abigail Spanberger"]
    all_candidates_research_data = []

    print("\nStep 1: Researching policy stances for each candidate...")
    for candidate_name in confirmed_candidates:
        print(f"\n--- Researching: {candidate_name} ---")
        candidate_data = []
        for policy_area in PREDEFINED_POLICY_AREAS:
            policy_research_response = await policy_research_executor.ainvoke({
                "candidate_name": candidate_name,
                "policy_area": policy_area,
                "query": f"Research the policy stances for {candidate_name} on the following policy area: \n {policy_area}."
            })
            candidate_research_obj = PydanticOutputParser(pydantic_object=PolicyStance).parse(
                policy_research_response["output"]
            )
            candidate_data.append(candidate_research_obj)
        all_candidates_research_data.append(candidate_data)

    print("\nStep 2: Compiling results into text file...")
    final_output_content = "Candidate Stances**\n\n"

    for i, candidate_data in enumerate(all_candidates_research_data):
        final_output_content += "---\n\n"
        final_output_content += f"**Candidate Name: {confirmed_candidates[i]}**\n\n"
        for j, policy_area in enumerate(PREDEFINED_POLICY_AREAS):
            final_output_content += f"**{policy_area}:**\n"
            final_output_content += f"{candidate_data[j].summary}\n"
            final_output_content += f"Source: {', '.join(candidate_data[j].sources)}\n\n"

    print(final_output_content)

    with open('output1.txt', 'w') as f:
        f.write(final_output_content)
    with open('research.pkl', 'wb') as f:
        pickle.dump(all_candidates_research_data, f)
    with open('candidates.pkl', 'wb') as f:
        pickle.dump(confirmed_candidates, f)
    with open('policyareas.pkl', 'wb') as f:
        pickle.dump(PREDEFINED_POLICY_AREAS, f)


if __name__ == "__main__":
    asyncio.run(run_research_workflow())
