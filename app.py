import os
import uuid

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, session
from flask_session import Session
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", os.urandom(32))
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

elections = [
    {"id": 1, "name": "2025 Virginia Gubernatorial Election"},
    {"id": 2, "name": "2025 NYC Mayoral Election"},
]


@app.route("/")
def election_selector():
    return render_template("elections.html", elections=elections)

@app.route("/chat/<id>")
def home(id):
    session.clear()
    return render_template("chat.html", id=id, elections_list=elections)

@app.route("/backend/<id>", methods=["POST"])
def chat(id):
    output_path = f"output/output{id}.txt"
    with open(output_path, "r", encoding="utf-8") as f:
        research_data = f.read()

    system_prompt = """You are an adaptive recommendation system. Your goal is to recommend a candidate for the mock election, given each candidate's stances on a variety of issues.

Candidate Research Data:
{all_research}

Instructions for Interaction:

Begin by asking the user one multiple-choice question related to practical, everyday life. Each answer choice must include a balanced explanation of pros and cons (about 3 sentences).

Continue asking one question at a time, selecting the next question based on what information would most help distinguish between candidates. Ask a maximum of 5 questions total.

Once you have enough information:
1. Provide a holistic review of how the user's preferences align with each candidate.
2. Summarize areas of alignment and tradeoffs in practical, everyday terms.
3. Do not give numeric scores; describe the fit in a way that helps the user make their own judgment.
4. Provide a recommendation to the user about who they align closest with."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]).partial(all_research=research_data)

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
        session_id = str(uuid.uuid4())
        session["sid"] = session_id

    try:
        response = with_message_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        return jsonify({"response": response["output"]})

    except Exception as e:
        return jsonify({"response": f"An error occurred: {e}. Please check your API key or server logs."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
