from flask import Flask, render_template, request, jsonify, session
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
import pickle
from flask_session import Session
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- Corrected Import for ChatGoogleGenerativeAI ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ChatMessageHistory
prefix = "/home/voteclear/mysite/"
prefix = ""
load_dotenv(dotenv_path=prefix+".env") #dotenv_path="/home/voteclear/mysite/.env"
class PolicyStance(BaseModel):
    summary: str = Field(description="A comprehensive summary of the candidate's stance.")
    sources: List[str] = Field(description="A list of full URLs to the verifiable sources.")
chat = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# with open('policyareas.pkl', 'rb') as f: PREDEFINED_POLICY_AREAS = pickle.load(f)
# with open('candidates.pkl', 'rb') as f: confirmed_candidates = pickle.load(f)
# with open('research.pkl', 'rb') as f: all_candidates_research_data = pickle.load(f)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0) # Using gemini-pro for its strong tool-calling capabilities. You can try "gemini-1.5-flash" or "gemini-1.5-pro" if you have access and prefer.


store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Wrap the agent_executor with RunnableWithMessageHistory
# This manages the chat history for each session

app = Flask(__name__)
app.secret_key = "afsdfashdfkshaedfkhasfd"  # required for Flask sessions
app.config["SESSION_TYPE"] = "filesystem"  # stores sessions locally
Session(app)
elections = [
        {"id": 1, "name": "2025 Virginia Gubernatorial Election"},
        {"id": 2, "name": "2025 NYC Mayoral Election"},
        # {"id": 3, "name": "Local School Board"},
        # {"id": 4, "name": "Mayor Election"}
    ]
# Use sessions (or just a fixed string) to track user interactions
@app.route("/")
def election_selector():
    # List of elections (you can fetch from DB/API later)
    return render_template("elections.html", elections=elections)

@app.route("/chat/<id>")
def home(id):
    session.clear()  
    return render_template("chat.html", id=id, elections_list=elections)  # HTML file to render chat UI

@app.route("/backend/<id>", methods=["POST"])
def chat(id):
    fn=prefix+"/output/output"+id+".txt"
    with open(fn, "r", encoding="utf-8") as f:
        final_output_content = f.read()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
            You are an adaptive recommendation system. Your goal is to recommend a candidate for the mock election, given each candidate’s stances on a variety of issues.
    
    Candidate Research Data:
    {all_research}

    Instructions for Interaction:

    You will search through the entire set of candidate data  each time you interact.

    Begin by asking the user one multiple-choice question related to practical, everyday life. The question should cover a real-world issue where people might have different preferences or priorities. Each question must be completely unique and explore a new dimension of daily tradeoffs, not something that could be inferred from answers to previous questions. Questions don’t only have to be about daily routines, but they must always be framed in ways that users can understand in practice rather than abstract or ideological terms (e.g., avoid statements like “I’d always support my country, whether it was right or wrong” or “If economic globalization is inevitable, it should primarily serve humanity rather than the interests of trans-national corporations”). Each answer choice must include a balanced explanation of pros and cons (about 3 sentences), showing how it could play out in everyday life.
    After the user responds, update your understanding of their values and preferences.

    Use that understanding to narrow down which candidate(s) may align more closely, but do not finalize a recommendation yet.

    Continue asking one question at a time, always selecting the next question based on what information would most help distinguish between candidates. Ask a maximum of 5 questions total; remember one at a time.

    Once you have enough information, stop asking questions and Instead do the following:
    1. Provide a holistic review of how the user’s preferences align with each candidate.
    2. Summarize areas of alignment and tradeoffs in practical, everyday terms.
    3. Do not give numeric scores; instead, describe the fit in a way that helps the user make their own judgment.
    4. Provide a recommendation to the user about who they align closest with.
                        """),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    ).partial(all_research = final_output_content)

    agent = create_tool_calling_agent(llm, [], prompt)
    agent_executor = AgentExecutor(agent=agent, tools=[], verbose=True)
    user_input = request.json.get("message", "")
    session_id = session.get("sid")
    with_message_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    if not session_id:
        # Assign a new session id
        import uuid
        session_id = str(uuid.uuid4())
        session["sid"] = session_id
    try:
        response = with_message_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        return jsonify({"response": response["output"]})

    except Exception as e:
        return jsonify({
            "response": f"An error occurred: {e}. Please check your API key or server logs."
        })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
