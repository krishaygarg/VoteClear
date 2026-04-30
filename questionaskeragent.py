from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

with open("Presidentialoutput.txt", "r", encoding="utf-8") as f:
    final_output_content = f.read()

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
]).partial(all_research=final_output_content)

agent = create_tool_calling_agent(llm, [], prompt)
agent_executor = AgentExecutor(agent=agent, tools=[], verbose=True)

store = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


with_message_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)


if __name__ == "__main__":
    session_id = "user_session_001"

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        try:
            response = with_message_history.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            print(f"Agent: {response['output']}")

        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            break