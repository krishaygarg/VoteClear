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
    # 1. Eleni Kounalakis (D)
    ek_id = db.save_candidate(ca_id, "Eleni Kounalakis", "Democrat", "https://eleniforca.com")
    db.save_policy_stance(ek_id, PREDEFINED_POLICY_AREAS[0], 
        "Eleni Kounalakis advocates for building middle-class wealth, focusing on making housing and higher education affordable. She supports targeted tax breaks for working families and investments in workforce training and high-wage job creation in technology and green energy.",
        ["https://eleniforca.com", "https://en.wikipedia.org/wiki/Eleni_Kounalakis"])
    db.save_policy_stance(ek_id, PREDEFINED_POLICY_AREAS[1],
        "As Lieutenant Governor and UC Regent, Kounalakis strongly supports expanding healthcare access, protecting reproductive rights, and increasing funding for California's public education systems (UC, CSU, and community colleges). She emphasizes addressing the homelessness crisis through state-supported housing projects and mental health services.",
        ["https://eleniforca.com", "https://en.wikipedia.org/wiki/Eleni_Kounalakis"])
    db.save_policy_stance(ek_id, PREDEFINED_POLICY_AREAS[2],
        "Kounalakis is a vocal supporter of California's aggressive climate mandates, advocating for a transition to 100% clean energy. She supports the expansion of offshore wind power, solar energy grids, and electric vehicle incentives, while pushing to phase out fossil fuel extraction in the state.",
        ["https://eleniforca.com", "https://en.wikipedia.org/wiki/Eleni_Kounalakis"])
    db.save_policy_stance(ek_id, PREDEFINED_POLICY_AREAS[3],
        "She supports trade partnerships that benefit California agriculture and technology exports. On immigration, she defends California's sanctuary state policies and advocates for humanitarian support, integration programs, and legal protections for undocumented residents.",
        ["https://eleniforca.com", "https://en.wikipedia.org/wiki/Eleni_Kounalakis"])

    # 2. Antonio Villaraigosa (D)
    av_id = db.save_candidate(ca_id, "Antonio Villaraigosa", "Democrat", "https://antoniovillaraigosa.com")
    db.save_policy_stance(av_id, PREDEFINED_POLICY_AREAS[0],
        "Antonio Villaraigosa focuses on economic growth through private-public partnerships, infrastructure spending, and red-tape reduction. Having served as Mayor of Los Angeles, he emphasizes job creation in transportation, construction, and tourism, while supporting fiscal responsibility and moderate tax policies to keep businesses in California.",
        ["https://antoniovillaraigosa.com", "https://en.wikipedia.org/wiki/Antonio_Villaraigosa"])
    db.save_policy_stance(av_id, PREDEFINED_POLICY_AREAS[1],
        "Villaraigosa places high priority on education reform, advocating for charter schools, teacher performance accountability, and vocational training. On homelessness, he favors combining strict enforcement of encampment bans with rapid building of temporary shelters and rehabilitation centers.",
        ["https://antoniovillaraigosa.com", "https://en.wikipedia.org/wiki/Antonio_Villaraigosa"])
    db.save_policy_stance(av_id, PREDEFINED_POLICY_AREAS[2],
        "He supports the transition to green energy but advocates for a pragmatically paced timeline to prevent energy grid blackouts and keep electricity bills affordable for families. He supports investments in water storage infrastructure to combat California droughts.",
        ["https://antoniovillaraigosa.com", "https://en.wikipedia.org/wiki/Antonio_Villaraigosa"])
    db.save_policy_stance(av_id, PREDEFINED_POLICY_AREAS[3],
        "He advocates for strong trade relations with Mexico and Pacific Rim countries to boost California ports. On immigration, he supports pathways to citizenship and protection of immigrant communities, combined with improved border management and security.",
        ["https://antoniovillaraigosa.com", "https://en.wikipedia.org/wiki/Antonio_Villaraigosa"])

    # 3. Chad Bianco (R)
    cb_id = db.save_candidate(ca_id, "Chad Bianco", "Republican", "https://chadbiancoforgovernor.com")
    db.save_policy_stance(cb_id, PREDEFINED_POLICY_AREAS[0],
        "Chad Bianco, current Riverside County Sheriff, advocates for lowering the cost of living by cutting state taxes and regulations. He criticizes California's current tax rates and business regulations as driving companies out of the state, and supports a free-market approach to job creation.",
        ["https://chadbiancoforgovernor.com", "https://en.wikipedia.org/wiki/Chad_Bianco"])
    db.save_policy_stance(cb_id, PREDEFINED_POLICY_AREAS[1],
        "Bianco's social policy centers heavily on law and order. He advocates for repealing Proposition 47 to enact harsher penalties for theft and drug crimes. On homelessness, he supports mandating mental health and drug rehabilitation programs and enforcing bans on street camping.",
        ["https://chadbiancoforgovernor.com", "https://en.wikipedia.org/wiki/Chad_Bianco"])
    db.save_policy_stance(cb_id, PREDEFINED_POLICY_AREAS[2],
        "Bianco opposes California's mandate to ban new gas-powered car sales by 2035 and calls for expanding domestic oil drilling and natural gas usage in California. He supports an 'all-of-the-above' energy strategy to lower utility rates for consumers.",
        ["https://chadbiancoforgovernor.com", "https://en.wikipedia.org/wiki/Chad_Bianco"])
    db.save_policy_stance(cb_id, PREDEFINED_POLICY_AREAS[3],
        "Bianco supports deploying the National Guard to the southern border to combat drug smuggling and illegal border crossings. He advocates for ending state-funded welfare programs for undocumented immigrants and opposes California's sanctuary state laws.",
        ["https://chadbiancoforgovernor.com", "https://en.wikipedia.org/wiki/Chad_Bianco"])


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
        "Hochul focuses on increasing public safety by amending bail reform laws, investing in mental health services, and building affordable housing units across NY. She supports funding for public schools, child care expansion, and defending reproductive healthcare rights.",
        ["https://kathyhochul.com", "https://en.wikipedia.org/wiki/Kathy_Hochul"])
    db.save_policy_stance(kh_id, PREDEFINED_POLICY_AREAS[2],
        "Hochul is committed to the Climate Leadership and Community Protection Act (CLCPA), aiming for 70% renewable electricity by 2030. She supports offshore wind, solar, and building electrification projects, while instituting a cap-and-invest program to reduce carbon emissions.",
        ["https://kathyhochul.com", "https://en.wikipedia.org/wiki/Kathy_Hochul"])
    db.save_policy_stance(kh_id, PREDEFINED_POLICY_AREAS[3],
        "She supports New York's status as a welcoming hub for immigrants but has called for federal assistance, expedited work permits, and border security coordination to manage the recent migrant housing and financial pressures in New York City.",
        ["https://kathyhochul.com", "https://en.wikipedia.org/wiki/Kathy_Hochul"])

    # 2. Lee Zeldin (R)
    lz_id = db.save_candidate(ny_id, "Lee Zeldin", "Republican", "https://leezeldin.com")
    db.save_policy_stance(lz_id, PREDEFINED_POLICY_AREAS[0],
        "Lee Zeldin advocates for cutting New York's income, property, and corporate taxes to combat the state's population loss. He supports slashing state spending, reforming the regulatory environment for businesses, and eliminating mandates that increase costs.",
        ["https://leezeldin.com", "https://en.wikipedia.org/wiki/Lee_Zeldin"])
    db.save_policy_stance(lz_id, PREDEFINED_POLICY_AREAS[1],
        "Zeldin emphasizes public safety, calling for the complete repeal of cashless bail laws and supporting police funding. On education, he is a strong advocate for school choice, expanding charter schools, and increasing parental oversight of curriculum content.",
        ["https://leezeldin.com", "https://en.wikipedia.org/wiki/Lee_Zeldin"])
    db.save_policy_stance(lz_id, PREDEFINED_POLICY_AREAS[2],
        "Zeldin advocates for lifting New York's ban on hydraulic fracturing (fracking) for natural gas in the Southern Tier to create jobs and lower heating bills. He opposes state mandates on electric vehicles and advocates for traditional fossil fuels alongside nuclear power.",
        ["https://leezeldin.com", "https://en.wikipedia.org/wiki/Lee_Zeldin"])
    db.save_policy_stance(lz_id, PREDEFINED_POLICY_AREAS[3],
        "Zeldin supports strict enforcement of immigration laws, calling for the repeal of New York's Green Light Law (which allows driver's licenses for undocumented immigrants) and ending sanctuary city policies in New York City.",
        ["https://leezeldin.com", "https://en.wikipedia.org/wiki/Lee_Zeldin"])

    print("Preloading complete! Wrote CA & NY 2026 elections.")

if __name__ == "__main__":
    reset_and_seed()
