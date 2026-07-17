import sqlite3
import os
import json

# DB is located in the project root (one level up from core/)
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "voteclear.db"))

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create elections table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS elections (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        date TEXT,
        ocd_id TEXT
    )
    """)
    
    # Create candidates table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS candidates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        election_id TEXT NOT NULL,
        name TEXT NOT NULL,
        party TEXT,
        website TEXT,
        FOREIGN KEY (election_id) REFERENCES elections (id)
    )
    """)
    
    # Create policy_stances table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS policy_stances (
        candidate_id INTEGER NOT NULL,
        policy_area TEXT NOT NULL,
        summary TEXT NOT NULL,
        sources TEXT,
        PRIMARY KEY (candidate_id, policy_area),
        FOREIGN KEY (candidate_id) REFERENCES candidates (id)
    )
    """)
    
    # Create conversations table for session storage
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        session_id TEXT PRIMARY KEY,
        election_id TEXT NOT NULL,
        history TEXT NOT NULL, -- JSON string of messages
        question_count INTEGER DEFAULT 0,
        FOREIGN KEY (election_id) REFERENCES elections (id)
    )
    """)
    
    conn.commit()
    conn.close()

def save_election(election_id, name, date=None, ocd_id=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO elections (id, name, date, ocd_id) VALUES (?, ?, ?, ?)",
        (str(election_id), name, date, ocd_id)
    )
    conn.commit()
    conn.close()

def save_candidate(election_id, name, party=None, website=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT id FROM candidates WHERE election_id = ? AND name = ?",
        (str(election_id), name)
    )
    row = cursor.fetchone()
    if row:
        candidate_id = row["id"]
        cursor.execute(
            "UPDATE candidates SET party = ?, website = ? WHERE id = ?",
            (party, website, candidate_id)
        )
    else:
        cursor.execute(
            "INSERT INTO candidates (election_id, name, party, website) VALUES (?, ?, ?, ?)",
            (str(election_id), name, party, website)
        )
        candidate_id = cursor.lastrowid
        
    conn.commit()
    conn.close()
    return candidate_id

def save_policy_stance(candidate_id, policy_area, summary, sources=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    sources_str = json.dumps(sources) if isinstance(sources, list) else sources
    cursor.execute(
        "INSERT OR REPLACE INTO policy_stances (candidate_id, policy_area, summary, sources) VALUES (?, ?, ?, ?)",
        (candidate_id, policy_area, summary, sources_str)
    )
    conn.commit()
    conn.close()

def get_all_elections():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, date FROM elections")
    rows = cursor.fetchall()
    conn.close()
    return [{"id": r["id"], "name": r["name"], "date": r["date"]} for r in rows]

def get_election(election_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM elections WHERE id = ?", (str(election_id),))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def get_candidates(election_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM candidates WHERE election_id = ?", (str(election_id),))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_policy_stances(candidate_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM policy_stances WHERE candidate_id = ?", (candidate_id,))
    rows = cursor.fetchall()
    conn.close()
    
    stances = []
    for r in rows:
        sources_list = []
        if r["sources"]:
            try:
                sources_list = json.loads(r["sources"])
            except json.JSONDecodeError:
                sources_list = [r["sources"]]
        stances.append({
            "policy_area": r["policy_area"],
            "summary": r["summary"],
            "sources": sources_list
        })
    return stances

def get_all_research_for_election(election_id):
    candidates = get_candidates(election_id)
    if not candidates:
        return ""
        
    output = "Candidate Stances**\n\n"
    for candidate in candidates:
        output += "---\n\n"
        output += f"**Candidate Name: {candidate['name']}**\n\n"
        
        stances = get_policy_stances(candidate["id"])
        for stance in stances:
            output += f"**{stance['policy_area']}:**\n"
            output += f"{stance['summary']}\n"
            source_text = ", ".join(stance["sources"]) if stance["sources"] else "None provided"
            output += f"Source: {source_text}\n\n"
            
    return output

def get_conversation(session_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM conversations WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "session_id": row["session_id"],
            "election_id": row["election_id"],
            "history": json.loads(row["history"]),
            "question_count": row["question_count"]
        }
    return None

def save_conversation(session_id, election_id, history_list, question_count=0):
    conn = get_db_connection()
    cursor = conn.cursor()
    history_json = json.dumps(history_list)
    cursor.execute(
        "INSERT OR REPLACE INTO conversations (session_id, election_id, history, question_count) VALUES (?, ?, ?, ?)",
        (session_id, str(election_id), history_json, question_count)
    )
    conn.commit()
    conn.close()

# Initialize tables on import
init_db()
