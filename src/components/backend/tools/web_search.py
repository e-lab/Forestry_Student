from langchain.tools import Tool
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.retrievers.web_research import WebResearchRetriever
from langchain_community.utilities import GoogleSearchAPIWrapper

class WebSearch: 
  def __init__(self, llm, vectorstore_public): 
    self.llm = llm 
    self.search = GoogleSearchAPIWrapper()
    self.web_retriever = WebResearchRetriever.from_llm(
        vectorstore=vectorstore_public,
        llm=self.llm, 
        search=self.search, 
        num_search_results=3
    )

  def initialize(self):
    return Tool.from_function(
        func=RetrievalQAWithSourcesChain.from_chain_type(llm=self.llm, retriever=self.web_retriever),
        name="web_QA",
        description="web_QA is a web searching tool for the LLM agent, triggered when the similarity score from in-context QA is too low. It dynamically integrates the LLM and a web retriever to broaden knowledge through targeted web searches, enhancing the agent's responsiveness and adaptability to diverse user queries",
    )