from components.backend.pipeline.vectorstore import VectorStore
from components.backend.pipeline.llm import LLM

import os, io 

from components.backend.tools.python_interpreter import PythonInterpreter
from components.backend.tools.arxiv_search import ArxivSearch
from components.backend.tools.calculator import Calculator
from components.backend.tools.web_search import WebSearch

from langchain.agents import initialize_agent

os.environ["OPENAI_API_KEY"] = "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"
os.environ['PINECONE_API_KEY'] = "204755b4-f7d8-4aa4-b16b-764e66796cc3"
os.environ["GOOGLE_API_KEY"] = "AIzaSyDKxAadUfBZ9oAMDlRjRe0jlp3N0oZKqvg"
os.environ["GOOGLE_CSE_ID"] = "57d010b1a25ce48c0"

class Pipeline: 
  def __init__(self, max_iterations=5): 
    self.llm = LLM()
    self.vectorstore = VectorStore() 
    self.tools = [
      PythonInterpreter(llm=self.llm.llm).initialize(),
      ArxivSearch().initialize(),
      Calculator(llm=self.llm.llm).initialize(),
      WebSearch(llm=self.llm.llm, vectorstore_public=self.vectorstore.vectorstore).initialize(),
    ]

    self.agent = initialize_agent(self.tools, 
      self.llm.llm, 
      agent="chat-conversational-react-description",
      verbose=True, 
      handle_parsing_errors=True, 
      max_iterations=max_iterations
    )
  
  def run(self, query, chat_history):
    return self.agent.invoke({'input': query.strip(), 'chat_history': chat_history}) 