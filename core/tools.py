import os
import json
import urllib.parse
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.tools import Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

load_dotenv()

def ddg_search(query: str, max_results: int = 3):
    """Scrapes DuckDuckGo HTML search results directly for a free lifetime search."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
    
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code != 200:
            return []
            
        soup = BeautifulSoup(res.text, "html.parser")
        results = []
        
        # Find all result snippets (which usually contain the main links)
        links = soup.find_all("a", class_="result__snippet")
        for link in links[:max_results]:
            snippet = link.get_text(strip=True)
            href = link.get("href", "")
            
            # Extract redirects if present
            parsed_url = href
            if "uddg=" in href:
                try:
                    parsed_url = urllib.parse.unquote(href.split("uddg=")[1].split("&")[0])
                except Exception:
                    pass
                
            parent = link.find_parent("div", class_="result")
            title = "No Title"
            if parent:
                title_el = parent.find("a", class_="result__a")
                if title_el:
                    title = title_el.get_text(strip=True)
                    
            results.append({
                "title": title,
                "url": parsed_url,
                "content": snippet
            })
            
        return results
    except Exception as e:
        print(f"Scraper Search Error: {e}")
        return []

def scrape_search_func(query: str) -> str:
    """Formatter to return search results as JSON string for LangChain agents."""
    results = ddg_search(query, max_results=3)
    if not results:
        return "No results found."
    return json.dumps(results, indent=2)

def save_to_txt(data: str, filename: str = "output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(formatted_text)
    return f"Data successfully saved to {filename}"

# Save tool
save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Save structured research data to a text file"
)

# Custom DuckDuckGo search tool replacement for Tavily
search_tool = Tool(
    name="web_search",
    func=scrape_search_func,
    description="A web search engine. Useful for looking up recent news, candidate stances, and facts about current events. Input should be a search query."
)

# Wikipedia tool
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=3000)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
