import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.tools import tool

from dotenv import load_dotenv
from duckduckgo_search import DDGS
from langchain.agents import Tool
from langchain.prompts import PromptTemplate
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup

from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup


from langchain.chains import LLMChain


GROQ_API_KEY = st.secrets['GROQ_API_KEY']


llm = ChatGroq(temperature=0.8, groq_api_key=GROQ_API_KEY, model_name="llama3-70b-8192", streaming = True)

## Arxiv and wikipedia Tools
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

DuckDuckGoSearchRun.requests_kwargs = {'verify': False}


# Set your OpenAI API key

@tool("ddg_search", return_direct=False)
def ddg_search_tool(query: str, num_results: int = 5) -> dict:
    """
    Perform a DuckDuckGo search and give results that user asks.

    Args:
        query (str): The search query.
        num_results (int): Number of search results to return.

    Returns:
        dict: Search results with optional crawled page content.
    """
   

    # Perform DuckDuckGo search
    results = DDGS(verify=False).text(query, max_results=5)
    if not results:
        return {"error": "No results found."}

    return(results)


    

search_tool = Tool(
    name="DuckDuckGo Search",
    func=ddg_search_tool,
    description="A search tool using DuckDuckGo to find information."
)



st.title("🔎 LangChain - Chat with search")
# """
# In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
# Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
# """

"""
In this example, we will see the working of an AI agent. Tyoe in your query and agent will display the thoughts and actions of an agent in an interactive Streamlit app.
"""



if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assisstant","content":"Hi,I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt:=st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)
    # search=DDGS(verify = False).text(prompt, max_results=10) 
    # llm=ChatGroq(groq_api_key=GROQ_API_KEY,model_name="Llama3-8b-8192",streaming=True)
    llm = ChatGroq(temperature=0.8, groq_api_key=GROQ_API_KEY, model_name="llama3-70b-8192", streaming = True)
    tools=[search_tool, arxiv, wiki]

    search_agent=initialize_agent(tools,llm,
                                  agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                  handling_parsing_errors=True, verbose = True)

    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({'role':'assistant',"content":response})
        st.write(response)

