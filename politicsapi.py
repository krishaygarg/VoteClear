import requests
import json
import re
# Replace with your actual API key obtained from Google Cloud Console
API_KEY = "AIzaSyC8LJWk0h4Bknna430G9szDk9ZCSIIttH8"


# A dictionary of sample addresses, keyed by state abbreviation.
# You MUST add more addresses here for states you want to query.
SAMPLE_ADDRESSES = {
    "CA": "1600 Amphitheatre Parkway, Mountain View, CA 94043",
    "NY": "1 Police Plaza, New York, NY 10038",
    "TX": "1100 Congress Ave, Austin, TX 78701",
    "FL": "400 S Monroe St, Tallahassee, FL 32399", # Residential-looking FL address
    "PA": "500 N 3rd St, Harrisburg, PA 17120",
    "MI": "2711 E. OUTER DR., Detroit, MI 48234", # Residential-type Michigan address
    # Add more state abbreviations and a valid residential-type address for each if you want to query more states
}

def get_elections(api_key):
    """Fetches a list of available elections from the Civic Information API."""
    elections_url = f"https://www.googleapis.com/civicinfo/v2/elections?key={api_key}"
    print("Fetching available elections...")
    try:
        response = requests.get(elections_url)
        response.raise_for_status()  # Raise an exception for HTTP errors (e.g., 403 Forbidden)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching elections: {e}")
        if e.response:
            print(f"Response content: {e.response.text}")
        return None

def get_voter_info(api_key, election_id, address):
    """
    Fetches voter information (including candidates) for a specific election and address.
    This version includes comprehensive error reporting by parsing the API's JSON response
    even if the HTTP status code is 200 OK but the API reports an internal error.
    """
    encoded_address = requests.utils.quote(address)
    voter_info_url = (
        f"https://www.googleapis.com/civicinfo/v2/voterinfo?"
        f"key={api_key}&address={encoded_address}&electionId={election_id}"
    )
    
    try:
        response = requests.get(voter_info_url)
        
        # --- Step 1: Try to parse the response as JSON regardless of HTTP status ---
        response_json = {}
        try:
            response_json = response.json()
        except json.JSONDecodeError:
            # If it's not JSON, it's likely a non-API error or a very unexpected response
            print(f"  Warning: Could not decode JSON from response for election ID {election_id} ({address}).")
            print(f"  HTTP Status: {response.status_code}, Raw content: {response.text}")
            return None # Indicate failure to get valid JSON

        # --- Step 2: Check the 'status' field within the JSON response ---
        # The Civic Information API often uses a 'status' field ("success" or "error")
        # in its JSON body, even if the HTTP status code is 200 OK.
        if response_json.get("status") == "success":
            return response_json # Successful API call and data found
        else:
            # The API itself reported an error or non-success status
            error_message = response_json.get("error", {}).get("message", "API reported non-success status.")
            print(f"  API Reported Status: '{response_json.get('status', 'N/A')}' for election ID {election_id} ({address}): {error_message}")
            
            # Print more specific error details if available in the 'errors' array
            if 'errors' in response_json.get('error', {}):
                for err in response_json['error']['errors']:
                    print(f"    - Detail: Reason: {err.get('reason', 'N/A')}, Message: {err.get('message', 'N/A')}")
            
            # Also print the HTTP status code if it indicates an error (e.g., 400 Bad Request)
            if response.status_code >= 400:
                print(f"  (HTTP Status Code: {response.status_code})")
            
            return None # Indicate API reported an error

    except requests.exceptions.RequestException as e:
        # --- Step 3: Handle network-level or request-specific errors ---
        # This catches errors like connection issues, timeouts, invalid URLs that don't get a response.
        print(f"  Network/Request Error for election ID {election_id} ({address}): {e}")
        if e.response is not None:
            print(f"  HTTP Status Code: {e.response.status_code}")
            print(f"  Raw response content: {e.response.text}") # Print raw content for non-JSON HTTP errors
        return None

def extract_state_from_ocd_division(ocd_division_id):
    """
    Extracts the US state abbreviation from an OCD Division ID.
    Example: ocd-division/country:us/state:ca -> CA
    Returns None if no state is found.
    """
    if ocd_division_id:
        match = re.search(r'state:([a-z]{2})', ocd_division_id)
        if match:
            return match.group(1).upper()
    return None

def main():
    """Main function to orchestrate fetching and displaying election candidates."""
    elections_data = get_elections(API_KEY)

    if not elections_data or "elections" not in elections_data:
        print("Could not retrieve elections or no elections available. Please check API key and network.")
        return

    print(f"\n--- Found {len(elections_data['elections'])} available elections ---")

    for election in elections_data["elections"]:
        election_id = election["id"]
        election_name = election["name"]
        election_day = election["electionDay"]
        ocd_division_id = election.get("ocdDivisionId") # OCD Division ID provides geographic scope

        print(f"\nProcessing Election: {election_name} (ID: {election_id}, Date: {election_day})")
        print(f"  OCD Division ID: {ocd_division_id}")

        # Attempt to determine the state of the election from its OCD Division ID
        election_state = extract_state_from_ocd_division(ocd_division_id)
        
        target_address = None
        if election_state and election_state in SAMPLE_ADDRESSES:
            target_address = SAMPLE_ADDRESSES[election_state]
            print(f"  Using address for {election_state}: {target_address}")
        elif ocd_division_id == "ocd-division/country:us":
            # For nationwide elections, use a default address (e.g., California) if available
            # This is a fallback if a more specific state isn't found/needed.
            if "CA" in SAMPLE_ADDRESSES:
                target_address = SAMPLE_ADDRESSES["CA"]
                print(f"  Using default address for nationwide election (CA): {target_address}")
            else:
                print(f"  Skipping: Nationwide election, but no default address (e.g., CA) in SAMPLE_ADDRESSES.")
                continue
        else:
            print(f"  Skipping: No matching sample address found for election's state or general division.")
            continue # Skip to the next election if no suitable address

        voter_info = get_voter_info(API_KEY, election_id, target_address)

        if voter_info and voter_info.get("status") == "success":
            if "contests" in voter_info:
                print("\n  Contests and Candidates:")
                for contest in voter_info["contests"]:
                    office_name = contest.get("office", "N/A")
                    district_name = contest.get("district", {}).get("name", "N/A")
                    print(f"    - Contest: {office_name} in {district_name}")

                    if "candidates" in contest:
                        for candidate in contest["candidates"]:
                            candidate_name = candidate.get("name", "N/A")
                            party = candidate.get("party", "N/A")
                            print(f"      - Candidate: {candidate_name} (Party: {party})")
                            if 'candidateUrl' in candidate:
                                print(f"        Website: {candidate['candidateUrl']}")
                            if 'channels' in candidate:
                                print("        Social Media:") # Added for clarity
                                for channel in candidate['channels']:
                                    print(f"          - Type: {channel.get('type', 'N/A')}, ID: {channel.get('id', 'N/A')}")
                    else:
                        print("      No candidates listed for this contest at this address.")
            else:
                print("    No contests found for this election at the provided address.")
        else:
            # Error message is already printed by get_voter_info if it failed
            print(f"    Could not retrieve valid voter info for this election using {target_address}.")


if __name__ == "__main__":
    main()
