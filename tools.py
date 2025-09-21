from langchain_core.tools import tool, BaseTool, Tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
duckduckgo_search_tool = DuckDuckGoSearchResults()
wikipedia_search_tool = Tool(name="wikipedia", func=WikipediaAPIWrapper(
).run, description="A tool to search Wikipedia for information on various topics.")
