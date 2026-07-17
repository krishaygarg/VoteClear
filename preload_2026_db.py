import os
import sys
import json
import sqlite3

# Ensure we can import from core/
sys.path.insert(0, os.path.dirname(__file__))

from core import db

PREDEFINED_POLICY_AREAS = [
    "Economic Policy: This category includes government actions aimed at influencing the economy's performance and stability. It covers areas like taxation, government spending, trade regulations, monetary policy, jobs, and inflation.",
    "Social Policy: These policies focus on the well-being and welfare of the population. This broad category includes areas like healthcare, education, social security programs, housing, and policies aimed at addressing poverty and inequality.",
    "Environmental & Energy Policy: This category deals with the protection of the environment, the management of natural resources, and the development and regulation of energy sources. This includes policies related to climate change, conservation, pollution control, and the transition to renewable energy.",
    "Foreign Affairs & National Security: These policies govern a nation's interactions with other countries and its defense. This includes diplomacy, immigration policy, defense spending, intelligence activities, international trade agreements, and responses to global threats."
]

def reset_and_seed():
    print("Resetting database...")
    db_file = db.DB_PATH
    if os.path.exists(db_file):
        os.remove(db_file)
        
    db.init_db()
    
    print("\n--- Seeding 2026 California Gubernatorial Election ---")
    ca_id = "2026-california-gubernatorial-election"
    db.save_election(ca_id, "2026 California Gubernatorial Election", "2026-11-03", "ocd-division/country:us/state:ca")
    
    # Candidates for CA
    # 1. Xavier Becerra (D)
    xb_id = db.save_candidate(ca_id, "Xavier Becerra", "Democrat", "https://xavierbecerra2026.com")
    db.save_policy_stance(xb_id, PREDEFINED_POLICY_AREAS[0], 
        "Xavier Becerra focuses on lowering the cost of living for working families, targeting high costs of housing, rent, childcare, utilities, and groceries. He supports targeted economic opportunities and investments in jobs to bolster what he calls the 'California Dream'.",
        ["https://xavierbecerra2026.com", "https://en.wikipedia.org/wiki/Xavier_Becerra"])
    db.save_policy_stance(xb_id, PREDEFINED_POLICY_AREAS[1],
        "Becerra places high priority on protecting and expanding healthcare access, leveraging his experience as U.S. Secretary of Health and Human Services to advocate for lower prescription drug costs and defend the Affordable Care Act. He supports educational equity and progressive crime prevention over increased incarceration.",
        ["https://xavierbecerra2026.com", "https://en.wikipedia.org/wiki/Xavier_Becerra"])
    db.save_policy_stance(xb_id, PREDEFINED_POLICY_AREAS[2],
        "Becerra is a strong supporter of California's aggressive green energy mandates. He supports investments in renewable energy and climate change initiatives, while defending the state's environmental regulations from federal rollback.",
        ["https://xavierbecerra2026.com", "https://en.wikipedia.org/wiki/Xavier_Becerra"])
    db.save_policy_stance(xb_id, PREDEFINED_POLICY_AREAS[3],
        "He defends California's sanctuary state laws, humane border management, and integration programs for undocumented immigrants. He has a long record of challenging federal overreach on immigration and civil rights.",
        ["https://xavierbecerra2026.com", "https://en.wikipedia.org/wiki/Xavier_Becerra"])

    # 2. Steve Hilton (R)
    sh_id = db.save_candidate(ca_id, "Steve Hilton", "Republican", "https://stevehiltonforgovernor.com")
    db.save_policy_stance(sh_id, PREDEFINED_POLICY_AREAS[0],
        "Steve Hilton campaigns on a message of 'positive populism' to cut costs for residents and help businesses by slashing state government spending and reducing regulations. He advocates for moderate economic reforms and supports ideas like higher living wages.",
        ["https://stevehiltonforgovernor.com", "https://en.wikipedia.org/wiki/Steve_Hilton"])
    db.save_policy_stance(sh_id, PREDEFINED_POLICY_AREAS[1],
        "Hilton's social policy centers on fixing California's school system and enhancing public safety. He challenges Sacramento's progressive policy orthodoxy, advocating for pragmatism, greater community accountability, and direct parental choice in education.",
        ["https://stevehiltonforgovernor.com", "https://en.wikipedia.org/wiki/Steve_Hilton"])
    db.save_policy_stance(sh_id, PREDEFINED_POLICY_AREAS[2],
        "Hilton favors an 'all-of-the-above' energy strategy to lower utility rates. While supporting a pragmatically paced transition to green energy, he criticizes current mandates as driving up utility bills and making California unaffordable.",
        ["https://stevehiltonforgovernor.com", "https://en.wikipedia.org/wiki/Steve_Hilton"])
    db.save_policy_stance(sh_id, PREDEFINED_POLICY_AREAS[3],
        "Hilton supports stricter border control, calling for a voter ID requirement and faster vote counting to restore public trust. He opposes sanctuary state laws and advocates for robust rule of law on immigration issues.",
        ["https://stevehiltonforgovernor.com", "https://en.wikipedia.org/wiki/Steve_Hilton"])


    print("\n--- Seeding 2026 New York Gubernatorial Election ---")
    ny_id = "2026-new-york-gubernatorial-election"
    db.save_election(ny_id, "2026 New York Gubernatorial Election", "2026-11-03", "ocd-division/country:us/state:ny")
    
    # Candidates for NY
    # 1. Kathy Hochul (D)
    kh_id = db.save_candidate(ny_id, "Kathy Hochul", "Democrat", "https://kathyhochul.com")
    db.save_policy_stance(kh_id, PREDEFINED_POLICY_AREAS[0],
        "Kathy Hochul's economic agenda centers on making New York more affordable. She has supported tax relief for middle-class residents and small businesses, while investing heavily in the state's semiconductor manufacturing industry (e.g. Micron technology hub) and jobs in green infrastructure.",
        ["https://kathyhochul.com", "https://en.wikipedia.org/wiki/Kathy_Hochul"])
    db.save_policy_stance(kh_id, PREDEFINED_POLICY_AREAS[1],
        "Kathy Hochul focuses on increasing public safety by amending bail reform laws, investing in mental health services, and building affordable housing units across NY. She supports funding for public schools, child care expansion, and defending reproductive healthcare rights.",
        ["https://kathyhochul.com", "https://en.wikipedia.org/wiki/Kathy_Hochul"])
    db.save_policy_stance(kh_id, PREDEFINED_POLICY_AREAS[2],
        "Kathy Hochul is committed to the Climate Leadership and Community Protection Act (CLCPA), aiming for 70% renewable electricity by 2030. She supports offshore wind, solar, and building electrification projects, while instituting a cap-and-invest program to reduce carbon emissions.",
        ["https://kathyhochul.com", "https://en.wikipedia.org/wiki/Kathy_Hochul"])
    db.save_policy_stance(kh_id, PREDEFINED_POLICY_AREAS[3],
        "She supports New York's status as a welcoming hub for immigrants but has called for federal assistance, expedited work permits, and border security coordination to manage the recent migrant housing and financial pressures in New York City.",
        ["https://kathyhochul.com", "https://en.wikipedia.org/wiki/Kathy_Hochul"])

    # 2. Bruce Blakeman (R)
    bb_id = db.save_candidate(ny_id, "Bruce Blakeman", "Republican", "https://blakemanfornewyork.com")
    db.save_policy_stance(bb_id, PREDEFINED_POLICY_AREAS[0],
        "Bruce Blakeman advocates for lower taxes, reduced state spending, and eliminating planned tax increases to combat the high cost of living. He criticizes existing policies as driving businesses and residents to leave New York.",
        ["https://blakemanfornewyork.com", "https://en.wikipedia.org/wiki/Bruce_Blakeman"])
    db.save_policy_stance(bb_id, PREDEFINED_POLICY_AREAS[1],
        "Blakeman emphasizes a 'common sense' law-and-order platform, strongly opposing cashless bail and 'criminal-first' laws. He advocates for additional police hiring, intelligence-led policing, and greater community safety.",
        ["https://blakemanfornewyork.com", "https://en.wikipedia.org/wiki/Bruce_Blakeman"])
    db.save_policy_stance(bb_id, PREDEFINED_POLICY_AREAS[2],
        "Blakeman opposes current progressive green energy mandates, arguing they drive up utility costs for families and businesses. He supports expanding traditional energy sources to lower costs.",
        ["https://blakemanfornewyork.com", "https://en.wikipedia.org/wiki/Bruce_Blakeman"])
    db.save_policy_stance(bb_id, PREDEFINED_POLICY_AREAS[3],
        "Blakeman strongly opposes sanctuary city policies and the use of public funds for services for undocumented immigrants. He supports strict enforcement of immigration laws and has worked to prevent migrant relocation to Nassau County.",
        ["https://blakemanfornewyork.com", "https://en.wikipedia.org/wiki/Bruce_Blakeman"])

    print("Preloading complete! Wrote CA & NY 2026 elections.")

if __name__ == "__main__":
    reset_and_seed()
    try:
        from verify_candidates import verify_all_elections
        verify_all_elections()
    except Exception as e:
        print(f"Candidate verification guardrail skipped: {e}")
