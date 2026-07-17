import os
import uuid
import asyncio
from typing import List, Sequence

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, session
from flask_session import Session
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI

from core import db
from core import election_manager
from core.tools import search_tool, wiki_tool

load_dotenv()

# Setup LLM - gemini-2.5-flash is active and supported
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

# Custom SQLite Message History class
class SQLiteChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str):
        self.session_id = session_id
        
    @property
    def messages(self) -> List[BaseMessage]:
        conv = db.get_conversation(self.session_id)
        if conv:
            return messages_from_dict(conv["history"])
        return []
        
    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        conv = db.get_conversation(self.session_id)
        election_id = conv["election_id"] if conv else "1"
        q_count = conv["question_count"] if conv else 0
        
        # Convert message list to dicts and save to DB
        history_dict = messages_to_dict(messages)
        db.save_conversation(self.session_id, election_id, history_dict, q_count)
        
    def clear(self) -> None:
        conv = db.get_conversation(self.session_id)
        if conv:
            db.save_conversation(self.session_id, conv["election_id"], [], 0)

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return SQLiteChatMessageHistory(session_id)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", os.urandom(32))
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

@app.route("/")
def election_selector():
    # Fetch elections dynamically from SQLite
    elections_list = db.get_all_elections()
    return render_template("elections.html", elections=elections_list)

@app.route("/chat/<id>")
def home(id):
    session.clear()
    
    # Initialize the session
    session_id = str(uuid.uuid4())
    session["sid"] = session_id
    
    # Create the conversation row in DB
    db.save_conversation(session_id, id, [], 0)
    
    # Get election info
    election_info = db.get_election(id)
    election_name = election_info["name"] if election_info else "Election"
    
    # Get candidates for sidebar
    candidates = db.get_candidates(id)
    
    return render_template("chat.html", id=id, election_name=election_name, candidates=candidates)

@app.route("/lookup-election", methods=["POST"])
def lookup_election():
    address = request.json.get("address", "").strip()
    if not address:
        return jsonify({"error": "Address is required"}), 400
        
    try:
        # Run the async load_or_create_election function in a synchronous context
        election_id = asyncio.run(election_manager.load_or_create_election(address))
        return jsonify({"election_id": election_id})
    except Exception as e:
        print(f"Error looking up election for address {address}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/backend/<id>", methods=["POST"])
def chat(id):
    # Retrieve candidates research from database
    research_data = db.get_all_research_for_election(id)

    system_prompt = """You are an adaptive political recommendation system. Your goal is to recommend a candidate for the election, given each candidate's stances on a variety of issues.

Candidate Research Data:
{all_research}

Instructions for Interaction:

1. Begin by asking the user one multiple-choice question related to practical, everyday life. Use A, B, C, D formatting for options, and each option must include a balanced explanation of pros and cons (about 3 sentences).
2. Continue asking one question at a time, selecting the next question based on what information would most help distinguish between candidates. Ask a maximum of 5 questions total.
3. Once you have enough information:
   - Provide a holistic review of how the user's preferences align with each candidate.
   - Summarize areas of alignment and tradeoffs in practical, everyday terms.
   - Do not give numeric scores; describe the fit in a way that helps the user make their own judgment.
   - Provide a recommendation to the user about who they align closest with.

If the user asks an out-of-quiz question or asks for candidate information during the conversation, use your search tools to look up real-time information to answer them accurately, then guide them back to the quiz."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]).partial(all_research=research_data)

    # Provide search and wikipedia lookup tools to the chatbot
    chat_tools = [search_tool, wiki_tool]
    
    agent = create_tool_calling_agent(llm, chat_tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=chat_tools, verbose=True)

    user_input = request.json.get("message", "")
    session_id = session.get("sid")

    # If session expired or is missing, regenerate it
    if not session_id:
        session_id = str(uuid.uuid4())
        session["sid"] = session_id
        db.save_conversation(session_id, id, [], 0)

    # Wrap the executor with SQL message history
    with_message_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    try:
        response = with_message_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        
        # Update question count in DB if applicable
        conv = db.get_conversation(session_id)
        if conv:
            # Simple check to see if agent asked a new question
            q_count = conv["question_count"]
            if "?" in response["output"] and q_count < 5:
                q_count += 1
                db.save_conversation(session_id, id, conv["history"], q_count)
                
        return jsonify({"response": response["output"]})

    except Exception as e:
        print(f"Chat execution error: {e}")
        return jsonify({"response": f"An error occurred: {e}. Please check your API key or server logs."})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5002))
    # Run database initialization
    db.init_db()
    app.run(host="0.0.0.0", port=port, debug=True)
