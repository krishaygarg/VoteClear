from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
import pickle
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
load_dotenv()
class PolicyStance(BaseModel):
    summary: str = Field(description="A comprehensive summary of the candidate's stance.")
    sources: List[str] = Field(description="A list of full URLs to the verifiable sources.")
chat = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# with open('policyareas.pkl', 'rb') as f: PREDEFINED_POLICY_AREAS = pickle.load(f)
# with open('candidates.pkl', 'rb') as f: confirmed_candidates = pickle.load(f)
# with open('research.pkl', 'rb') as f: all_candidates_research_data = pickle.load(f)
# final_output_content = "Candidate Stances**\n\n"

# for i in range(len(all_candidates_research_data)):
#     final_output_content += "---\n\n"
#     final_output_content += f"**Candidate Name: {confirmed_candidates[i]}**\n\n"
#     for j in range(len(PREDEFINED_POLICY_AREAS)):
#         final_output_content += f"**{PREDEFINED_POLICY_AREAS[j]}:**\n"
#         final_output_content += f"{all_candidates_research_data[i][j].summary}\n"
#         final_output_content += f"Source: {', '.join(all_candidates_research_data[i][j].sources)}\n\n"

with open("Presidentialoutput.txt", "r", encoding="utf-8") as f:
    final_output_content = f.read()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0) # Using gemini-pro for its strong tool-calling capabilities. You can try "gemini-1.5-flash" or "gemini-1.5-pro" if you have access and prefer.

# Define the prompt for the agent
# The system message guides the agent's behavior
# MessagesPlaceholder("chat_history") is crucial for memory
# MessagesPlaceholder("agent_scratchpad") is where the agent's thoughts and tool outputs go
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

# Create the agent
# create_tool_calling_agent is excellent for agents that primarily interact via tools
agent = create_tool_calling_agent(llm, [], prompt)

# Create the AgentExecutor
# This is the runtime for the agent, managing its steps (reasoning, tool calls, responses)
agent_executor = AgentExecutor(agent=agent, tools=[], verbose=True)

# --- 4. Add Conversational Memory ---
# This ensures the agent remembers the conversation history.
# We'll use a simple in-memory history for demonstration.
# In a real app, you'd store this in a database (e.g., Redis, PostgreSQL, etc.)
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Wrap the agent_executor with RunnableWithMessageHistory
# This manages the chat history for each session
with_message_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# --- 5. Interact with the Adaptive Recommendation System ---

if __name__ == "__main__":
    print("\n" + "="*50)
    print("="*50 + "\n")

    session_id = "user123_gemini_session" # A unique ID for the user's session

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        try:
            print("\n--- Agent's Internal Process ---")
            # Invoke the agent with the user's input and the session ID
            # The 'config' parameter is where you pass the session_id
            response = with_message_history.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            print("--- End of Agent's Internal Process ---")
            print(f"Agent: {response['output']}")

    
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            print("Please check the traceback above for details.")
            print("Ensure your Gemini API key is correct and valid.")
            # Optionally, you can add more sophisticated error handling or fallback responses here.
            break # Exit on unhandled error