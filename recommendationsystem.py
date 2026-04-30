import os
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

load_dotenv()

google_api_key = os.getenv('GOOGLE_API_KEY')
gemini_api_key = os.getenv('GEMINI_API_KEY')

if not google_api_key and not gemini_api_key:
    print("\nCRITICAL ERROR: Neither GOOGLE_API_KEY nor GEMINI_API_KEY found in environment.")
    print("Please ensure your .env file is correctly configured.")
    print("You can get a Gemini API key from https://ai.google.dev/gemini-api/docs/api-key")
    sys.exit(1)


RECOMMENDATION_DATA = {
    "movies": [
        {"title": "Inception", "genre": "Sci-Fi", "year": 2010, "rating": 8.8, "plot": "A thief who steals corporate secrets through use of dream-sharing technology."},
        {"title": "The Matrix", "genre": "Sci-Fi", "year": 1999, "rating": 8.7, "plot": "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers."},
        {"title": "Pulp Fiction", "genre": "Crime", "year": 1994, "rating": 8.9, "plot": "The lives of two mob hitmen, a boxer, a gangster's wife, and a pair of diner bandits intertwine in four tales of violence and redemption."},
        {"title": "Forrest Gump", "genre": "Drama", "year": 1994, "rating": 8.8, "plot": "The presidencies of Kennedy and Johnson, the Vietnam War, the Watergate scandal and other historical events unfold from the perspective of an Alabama man with an IQ of 75."},
        {"title": "Spirited Away", "genre": "Animation", "year": 2001, "rating": 8.6, "plot": "During her family's move to the suburbs, a sullen 10-year-old girl wanders into a world ruled by gods, witches, and spirits, and where humans are changed into beasts."},
        {"title": "The Lord of the Rings: The Fellowship of the Ring", "genre": "Fantasy", "year": 2001, "rating": 8.8, "plot": "A young hobbit, Frodo Baggins, inherits a magical ring that he must destroy to save Middle-earth from the Dark Lord Sauron."},
        {"title": "Blade Runner 2049", "genre": "Sci-Fi", "year": 2017, "rating": 8.0, "plot": "Young Blade Runner K discovers a long-buried secret that has the potential to plunge what's left of society into chaos."},
        {"title": "Interstellar", "genre": "Sci-Fi", "year": 2014, "rating": 8.6, "plot": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival."},
        {"title": "Parasite", "genre": "Thriller", "year": 2019, "rating": 8.5, "plot": "Greed and class discrimination threaten the newly formed symbiotic relationship between the wealthy Park family and the destitute Kim family."},
    ],
    "books": [
        {"title": "Dune", "genre": "Sci-Fi", "author": "Frank Herbert", "year": 1965, "rating": 8.7},
        {"title": "1984", "genre": "Dystopian", "author": "George Orwell", "year": 1949, "rating": 8.8},
        {"title": "The Hobbit", "genre": "Fantasy", "author": "J.R.R. Tolkien", "year": 1937, "rating": 8.5},
        {"title": "Pride and Prejudice", "genre": "Romance", "author": "Jane Austen", "year": 1813, "rating": 8.0},
        {"title": "To Kill a Mockingbird", "genre": "Classic", "author": "Harper Lee", "year": 1960, "rating": 8.3},
    ]
}

# --- 2. Define Tools for the Agent ---
# Tools are functions the LLM can call to perform specific actions.

class MoviePreference(BaseModel):
    genre: Optional[str] = Field(None,description="The preferred genre of the movie.")
    year_range: Optional[str] = Field(None,description="A specified year range (e.g., '2000s', 'recent', '1990-2010').")
    min_rating: Optional[float] = Field(None,description="Minimum IMDb rating preference (0.0 to 10.0).")

class BookPreference(BaseModel):
    genre: Optional[str] = Field(None,description="The preferred genre of the book.")
    author: Optional[str]= Field(None,description="The preferred author of the book (optional).")
    year_range: Optional[str] = Field(None,description="A specified year range for the book's publication (e.g., 'recent', 'pre-2000').")
global finished 
finished = False
@tool
def finish_run():
    """
    Finishes the duties of the AI agent.
    """
    global finished
    finished = True
@tool
def get_movie_recommendations(preferences: MoviePreference) -> List[Dict[str, Any]]:
    """Provides movie recommendations based on user preferences."""
    results = []
    for movie in RECOMMENDATION_DATA["movies"]:
        match_genre = preferences.genre.lower() in movie["genre"].lower() if preferences.genre else True
        match_rating = movie["rating"] >= preferences.min_rating if preferences.min_rating else True

        match_year = True
        if preferences.year_range:
            try:
                if "recent" in preferences.year_range.lower():
                    match_year = movie["year"] >= 2020
                elif "-" in preferences.year_range:
                    start_year, end_year = map(int, preferences.year_range.split('-'))
                    match_year = start_year <= movie["year"] <= end_year
                elif "s" in preferences.year_range: # e.g., 1990s
                    decade_start = int(preferences.year_range.replace('s', ''))
                    match_year = decade_start <= movie["year"] < decade_start + 10
            except ValueError:
                pass # Malformed year_range, ignore for now

        if match_genre and match_rating and match_year:
            results.append(movie)
    return results if results else ["No movies found matching these preferences. Please try adjusting your criteria."]

@tool
def get_book_recommendations(preferences: BookPreference) -> List[Dict[str, Any]]:
    """Provides book recommendations based on user preferences."""
    results = []
    for book in RECOMMENDATION_DATA["books"]:
        match_genre = preferences.genre.lower() in book["genre"].lower() if preferences.genre else True
        match_author = preferences.author.lower() in book["author"].lower() if preferences.author else True

        match_year = True
        if preferences.year_range:
            try:
                if "recent" in preferences.year_range.lower():
                    match_year = book["year"] >= 2010
                elif "pre-" in preferences.year_range.lower():
                    year_limit = int(preferences.year_range.split('-')[1])
                    match_year = book["year"] < year_limit
                elif "-" in preferences.year_range:
                    start_year, end_year = map(int, preferences.year_range.split('-'))
                    match_year = start_year <= book["year"] <= end_year
            except ValueError:
                pass # Malformed year_range, ignore for now

        if match_genre and match_author and match_year:
            results.append(book)
    return results if results else ["No books found matching these preferences. Please try adjusting your criteria."]

@tool
def clarify_preference_type(query: str) -> str:
    """Asks the user to clarify if they are looking for movie or book recommendations."""
    return "Are you looking for movie recommendations or book recommendations?"


tools = [get_movie_recommendations, get_book_recommendations, clarify_preference_type, finish_run]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an adaptive recommendation system. Your goal is to recommend movies or books to the user. "
        "You need to ask clarifying questions to understand their preferences. "
        "If the user's request is ambiguous, ask them to specify if they want movies or books. "
        "Once you have enough information, use the appropriate tool to find recommendations. "
        "If no recommendations are found, politely inform the user and ask if they'd like to adjust their criteria."
        "Once you are done giving a recommendation, call the finish_run tool."
        "\n\nBegin by asking about their general preference (movies or books)."
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

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

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Welcome to the Adaptive Recommendation System!")
    print("I can recommend movies or books. Type 'exit' to quit.")
    print("="*50 + "\n")

    session_id = "user_session_001"

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        if finished:
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