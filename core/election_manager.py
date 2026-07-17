import os
import requests
import json
import asyncio
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Relative imports from current package
from . import db
from . import research

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

class ElectionDiscovery(BaseModel):
    election_name: str = Field(description="Name of the most prominent upcoming state or local election for the address (e.g. 2026 California Gubernatorial Election).")
    candidates: list[str] = Field(description="List of the confirmed major candidate names running in this election.")

election_discovery_parser = PydanticOutputParser(pydantic_object=ElectionDiscovery)

async def discover_election_via_llm(address: str) -> dict:
    """Uses LLM to identify the prominent upcoming election and candidates for an address as a fallback."""
    print(f"Using LLM to discover election for address: {address}...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert political advisor. Given a user's address, your job is to identify the most prominent upcoming local or state election (such as Gubernatorial, Mayoral, or Senate election in 2026/2027) for that location and list the major confirmed candidates.
        
        Focus on real, upcoming major elections (e.g., if NY, then New York Gubernatorial; if CA, then California Gubernatorial; if TX, then Texas Gubernatorial).
        Provide your response as a JSON object matching this schema:
        {format_instructions}"""),
        ("human", "Address: {address}")
    ]).partial(format_instructions=election_discovery_parser.get_format_instructions())
    
    try:
        res = await llm.ainvoke(prompt.format(address=address))
        parsed = election_discovery_parser.parse(res.content)
        return {
            "election_name": parsed.election_name,
            "candidates": parsed.candidates
        }
    except Exception as e:
        print(f"Error discovering election via LLM: {e}")
        return None

def fetch_civic_voter_info(address: str):
    if not API_KEY:
        return None
        
    encoded_address = requests.utils.quote(address)
    url = f"https://www.googleapis.com/civicinfo/v2/voterinfo?key={API_KEY}&address={encoded_address}"
    try:
        res = requests.get(url)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        print(f"Civic API error: {e}")
    return None

async def load_or_create_election(address: str) -> str:
    """
    Given an address:
    1. Parses address to find the corresponding election and candidates.
    2. Checks if election already exists in database.
    3. If not, saves it, triggers dynamic research for each candidate, and saves stances.
    4. Returns the election ID.
    """
    civic_data = fetch_civic_voter_info(address)
    election_name = None
    candidates = []
    
    if civic_data and "contests" in civic_data:
        for contest in civic_data["contests"]:
            office = contest.get("office", "").lower()
            if "governor" in office or "mayor" in office or "senate" in office or "president" in office:
                election_name = f"Upcoming {contest.get('office')} Election"
                if "candidates" in contest:
                    candidates = [c.get("name") for c in contest["candidates"]]
                break
                
    if not election_name or not candidates:
        llm_discovered = await discover_election_via_llm(address)
        if llm_discovered:
            election_name = llm_discovered["election_name"]
            candidates = llm_discovered["candidates"]
            
    if not election_name or not candidates:
        raise ValueError("Could not identify an election or candidates for the provided address.")
        
    # Generate a unique slug ID for the election
    election_id = election_name.lower().replace(" ", "-").replace("'", "").replace('"', "")
    election_id = re.sub(r'[^a-z0-9\-]', '', election_id)
    
    existing = db.get_election(election_id)
    if existing:
        print(f"Election '{election_name}' (ID: {election_id}) already exists in the database. Loading...")
        return election_id
        
    print(f"Creating new election: {election_name} (ID: {election_id})")
    db.save_election(election_id, election_name)
    
    for cand_name in candidates:
        await research.research_candidate_and_save(election_id, cand_name)
        
    # Generate compatibility text file
    output_content = db.get_all_research_for_election(election_id)
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"output{election_id}.txt"), "w", encoding="utf-8") as f:
        f.write(output_content)
        
    print(f"Successfully researched and loaded election: {election_name}")
    return election_id
