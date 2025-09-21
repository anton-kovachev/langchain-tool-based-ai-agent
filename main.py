from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from langchain.agents import AgentExecutor, create_tool_calling_agent
from tools import duckduckgo_search_tool, wikipedia_search_tool

load_dotenv()


class ReplyInstructions(BaseModel):
    reply: str
    sources: list[str]
    tools: list[str]


parser = PydanticOutputParser(pydantic_object=ReplyInstructions)
anthropic_llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
# openai_llm = OpenAI(model_name="gpt-4", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("placeholder", "{chat_history}"),
    ("system",
     "You are a helpful scientific research that provides information for the asked questions in the following format {format_instructions}."),
    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}")
]).partial(format_instructions=parser.get_format_instructions())


anthropic_agent = create_tool_calling_agent(
    llm=anthropic_llm,
    tools=[wikipedia_search_tool, duckduckgo_search_tool],
    prompt=prompt
)

anthropic_agent_executor = AgentExecutor(
    agent=anthropic_agent, tools=[wikipedia_search_tool, duckduckgo_search_tool], verbose=True)

# chain = prompt | anthropic_llm | parser
if __name__ == "__main__":
    while True:
        query = input("Enter your query: ")
        if query in {"quit", "exit"}:
            break

        # reply = chain.invoke(
        #     {"query": query})
        # print(reply)
        raw_reply = anthropic_agent_executor.invoke(
            {"query": query})

        # Parse the agent's output using the PydanticOutputParser
        try:
            parsed_response = parser.parse(
                raw_reply.get("output", "")[0]["text"])
            print(f"Reply: {parsed_response.reply}")
            print(f"Sources: {parsed_response.sources}")
            print(f"Tools used: {parsed_response.tools}")
        except Exception as e:
            print(f"Error parsing response: {e}")
            print(f"Raw output: {raw_reply.get('output', '')}")
