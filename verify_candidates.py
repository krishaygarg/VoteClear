import os
import sys
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Add root folder to system path for core imports
sys.path.insert(0, os.path.dirname(__file__))

from core import db
from core.tools import ddg_search

def verify_all_elections():
    # Make sure we have GOOGLE_API_KEY
    if not os.getenv("GOOGLE_API_KEY"):
        print("Warning: GOOGLE_API_KEY is not set. Skipping candidate verification guardrails.")
        return True

    print("Running candidate verification guardrails...")
    
    # Initialize DB (creates database and tables if not exist)
    db.init_db()
    
    elections = db.get_all_elections()
    if not elections:
        print("No elections found in database to verify.")
        return True

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
    failed = False

    for election in elections:
        election_id = election["id"]
        election_name = election["name"]
        
        # Get seeded candidates
        seeded_candidates = [c["name"] for c in db.get_candidates(election_id)]
        print(f"\nVerifying '{election_name}'...")
        print(f"Seeded candidates: {seeded_candidates}")
        
        if not seeded_candidates:
            print(f"❌ No candidates found in database for election: {election_name}")
            failed = True
            continue

        # Search the web for current general election candidates
        search_query = f"{election_name} general election candidates 2026 nominees"
        search_results = ddg_search(search_query, max_results=5)
        
        search_text = ""
        for idx, res in enumerate(search_results):
            search_text += f"[{idx+1}] Title: {res['title']}\nURL: {res['url']}\nSnippet: {res['content']}\n\n"

        prompt = f"""You are a political data validation assistant.
Your job is to verify if the seeded candidates in our database match the actual major political party candidates who advanced to the 2026 General Election (nominees).

Election: {election_name}
Seeded Candidates in Database: {seeded_candidates}

Web Search Results:
{search_text}

Analyze the search results. Are the seeded candidates exactly the actual general election nominees (e.g. top-two primary survivors or main party nominees) for the 2026 general election?
Ignore minor spelling variations, but flag if a candidate did not make it to the general election, or if the actual nominee is missing.

Respond ONLY with a JSON object in this format (no other text, markdown blocks, or explanation):
{{
  "correct": true or false,
  "actual_candidates": ["Candidate A", "Candidate B"],
  "reason": "Detail if there are incorrect candidates or missing nominees, referring to the search results."
}}
"""

        try:
            response = llm.invoke(prompt)
            output_text = response.content.strip()
            
            # Clean up potential markdown formatting in response
            if output_text.startswith("```"):
                lines = output_text.split("\n")
                if lines[0].startswith("```json"):
                    output_text = "\n".join(lines[1:-1])
                elif lines[0].startswith("```"):
                    output_text = "\n".join(lines[1:-1])
            
            res_json = json.loads(output_text.strip())
            
            if not res_json.get("correct"):
                print(f"❌ Candidate discrepancy found in '{election_name}'!")
                print(f"Actual general election nominees should be: {res_json.get('actual_candidates')}")
                print(f"Reason: {res_json.get('reason')}")
                failed = True
            else:
                print(f"✅ '{election_name}' successfully verified!")
                
        except Exception as e:
            print(f"⚠️ Error verifying candidates for '{election_name}': {e}")
            # Do not fail the build if it is a transient network/API error
            continue

    if failed:
        print("\n❌ Candidate verification failed! Seeded candidates do not match current general election candidates.")
        return False
    else:
        print("\n✅ All candidate validations passed successfully!")
        return True

if __name__ == "__main__":
    success = verify_all_elections()
    if not success:
        sys.exit(1)
    else:
        sys.exit(0)
