import os
from typing import List, Dict, Any, Union
from dotenv import load_dotenv
from typing import Optional
# --- Load environment variables FIRST ---
# This ensures GOOGLE_API_KEY is available when ChatGoogleGenerativeAI is instantiated.
load_dotenv()

# --- Debugging API Key Loading ---
print(f"Current working directory: {os.getcwd()}")
dotenv_path = os.path.join(os.getcwd(), '.env')
print(f"Looking for .env at: {dotenv_path}")
print(f".env file exists at path: {os.path.exists(dotenv_path)}")

google_api_key_loaded = os.getenv('GOOGLE_API_KEY')
gemini_api_key_loaded = os.getenv('GEMINI_API_KEY') # Check for both as a fallback

if google_api_key_loaded:
    print(f"GOOGLE_API_KEY environment variable: {'***** (found)'}")
elif gemini_api_key_loaded:
    print(f"GEMINI_API_KEY environment variable: {'***** (found - using this as fallback)'}")
else:
    print("\nCRITICAL ERROR: Neither GOOGLE_API_KEY nor GEMINI_API_KEY found in environment.")
    print("Please ensure your .env file is correctly configured and located in the current directory.")
    print("Example .env content: GOOGLE_API_KEY=\"YOUR_ACTUAL_GEMINI_API_KEY\"")
    print("You can get a Gemini API key from https://ai.google.dev/gemini-api/docs/api-key")
    import sys
    sys.exit(1) # Exit if no API key is found

# --- LangChain Imports ---
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- Corrected Import for ChatGoogleGenerativeAI ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ChatMessageHistory


# --- 1. Define your Recommendation Data (Simulated for this example) ---
# In a real system, this would come from a database, API, etc.
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
    """
    Provides movie recommendations based on user preferences.
    Takes genre, year_range, and min_rating as input.
    Example: get_movie_recommendations({'genre': 'Sci-Fi', 'year_range': '2010s', 'min_rating': 8.5})
    """
    print(f"\n--- Tool Called: get_movie_recommendations with preferences: {preferences} ---\n")
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
    """
    Provides book recommendations based on user preferences.
    Takes genre, author (optional), and year_range as input.
    Example: get_book_recommendations({'genre': 'Fantasy', 'author': 'J.R.R. Tolkien', 'year_range': 'pre-1950'})
    """
    print(f"\n--- Tool Called: get_book_recommendations with preferences: {preferences} ---\n")
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

# A tool to confirm user preference type (movie/book)
@tool
def clarify_preference_type(query: str) -> str:
    """
    Asks the user to clarify if they are looking for movie or book recommendations if it's unclear.
    The 'query' parameter should be the user's ambiguous request.
    Example: clarify_preference_type("I want something good to watch.")
    """
    print(f"\n--- Tool Called: clarify_preference_type with query: {query} ---\n")
    return "Are you looking for movie recommendations or book recommendations?"


tools = [get_movie_recommendations, get_book_recommendations, clarify_preference_type, finish_run]

# --- 3. Define the LLM and Agent ---
# Ensure GOOGLE_API_KEY is set as an environment variable (loaded by load_dotenv())
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0) # Using gemini-pro for its strong tool-calling capabilities. You can try "gemini-1.5-flash" or "gemini-1.5-pro" if you have access and prefer.

# Define the prompt for the agent
# The system message guides the agent's behavior
# MessagesPlaceholder("chat_history") is crucial for memory
# MessagesPlaceholder("agent_scratchpad") is where the agent's thoughts and tool outputs go
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an adaptive recommendation system. Your goal is to recommend movies or books to the user. "
            "You need to ask clarifying questions to understand their preferences (e.g., genre, year, rating for movies; genre, author, year for books). "
            "If the user's request is ambiguous (e.g., 'recommend something'), you must ask them to specify if they want movies or books. "
            "Once you have enough information, use the appropriate tool to find recommendations. "
            "If no recommendations are found, politely inform the user and ask if they'd like to adjust their criteria."
            "Once you are done giving a recommendation, call the appropriate tool to finish your run."
            "\n\nBegin by asking about their general preference (movies or books)."
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

# Create the agent
# create_tool_calling_agent is excellent for agents that primarily interact via tools
agent = create_tool_calling_agent(llm, tools, prompt)

# Create the AgentExecutor
# This is the runtime for the agent, managing its steps (reasoning, tool calls, responses)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

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
    print("Welcome to the Adaptive Recommendation System (powered by Gemini)!")
    print("I can recommend movies or books. What are you looking for today?")
    print("Type 'exit' to quit.")
    print("="*50 + "\n")

    session_id = "user123_gemini_session" # A unique ID for the user's session

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        if finished == True:
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