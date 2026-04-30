from datetime import datetime

from dotenv import load_dotenv
from langchain.tools import Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import WikipediaAPIWrapper

load_dotenv()


def save_to_txt(data: str, filename: str = "output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(formatted_text)
    return f"Data successfully saved to {filename}"


save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Save structured research data to a text file"
)

search_tool = TavilySearchResults(max_results=3)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=3000)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)


