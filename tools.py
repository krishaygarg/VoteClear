import pprint
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper, SearxSearchWrapper
from langchain.tools import Tool
from langchain_community.tools.searx_search.tool import SearxSearchResults
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import certifi
load_dotenv()

from datetime import datetime
def save_to_txt(data: str, filename: str= "output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%m:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"


    with open(filename, "w",encoding="utf-8") as f:
        f.write(formatted_text)
    return f"Data successfully saved to {filename}"


save_tool = Tool(
    name = "save_text_to_file",
    func = save_to_txt,
    description = "Save structured research data to a text file"
)
#search = DuckDuckGoSearchRun()
wrapper  = SearxSearchWrapper(searx_host="http://127.0.0.1:8080")
#search_tool = SearxSearchResults(name="Google", wrapper=wrapper,kwargs = {"engines": ["google"]})
# search_tool = Tool(
#     name="search",
#     func=search.run,
#     description="Search the web for information using SearXNG"
# )
search_tool = TavilySearchResults(max_results=3)


api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=3000)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)


