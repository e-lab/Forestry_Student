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
        description="web_QA is a web searching tool for the LLM agent. Use this only after you have tried the VectorDB tool. FIRST, TRY THE VectorDB TOOL.",
    )