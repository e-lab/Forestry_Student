from components.backend.pipeline.vectorstore import VectorStore
from components.backend.pipeline.llm import LLM
from components.backend.pipeline.document_handler import Document_Handler
from langchain_community.vectorstores import Chroma
import os, io 

from components.backend.tools.python_interpreter import PythonInterpreter
from components.backend.tools.arxiv_search import ArxivSearch
from components.backend.tools.calculator import Calculator
from components.backend.tools.web_search import WebSearch
from components.backend.tools.in_context_qa import InContextQA
from components.backend.tools.csv_agent import CSVAgent
import streamlit as st 
from langchain.agents import initialize_agent, create_openai_tools_agent
from langchain.agents import AgentExecutor
from langchain import hub
import time
import threading
import re 

def timeout(limit=90):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = [None] 

            def target():
                result[0] = func(*args, **kwargs)

            thread = threading.Thread(target=target)
            thread.start()
            thread.join(limit)  

            if thread.is_alive():
                print("Function execution exceeded 1.5 minutes, returning None.")
                thread.join()  
                return None
            return result[0]
        return wrapper
    return decorator

class Pipeline: 
  def __init__(self, max_iterations=5): 
    self.document_handler = Document_Handler() 
    self.max_iterations = max_iterations

  @timeout(limit=90)
  def run_normal(self, query, chat_history, csv_paths=[], pdf_paths=[]):
    
    llm = LLM()
    vectorstore = VectorStore() 

    tools = [
      PythonInterpreter(llm=llm.llm).initialize(),
      ArxivSearch().initialize(),
      Calculator(llm=llm.llm).initialize(),
      WebSearch(llm=llm.llm, vectorstore_public=vectorstore.vectorstore).initialize(),
    ]
    if pdf_paths: 
      tools += [InContextQA(vectorstore=vectorstore).initialize(pdf_paths, self.document_handler)]
    if csv_paths:
      tools += [CSVAgent(llm.llm, csv_paths).initialize()]

    tools_agent = create_openai_tools_agent(llm.llm, 
            tools,
            hub.pull("hwchase17/openai-functions-agent"),
          )

    tools_agent = AgentExecutor(agent=tools_agent, tools=tools, verbose=True, handle_parsing_errors=True, 
            max_iterations=self.max_iterations)

    result = tools_agent.invoke({'input': f"""
    <system> 
    IF YOU ARE ASKED ABOUT PEOPLE or SCIENTIFIC TERMS: START WITH in_context_qa.
    FOR UNKNOWN NOUNS, NAMES AND OBJECTS: START WITH in_context_qa, THEN arxiv_search.
    FOR QUESTIONS ABOUT DATA: START WITH csv_agent.
    IF YOU ARE ASKED TO VISUALIZE DATA: START WITH csv_agent, ASK IT TO MAKE A DF QUERYING COMMAND, PASS IT TO python_interpreter TO USE THE QUERY TO SAVE A VISUALIZATION. In this case, return ONLY "||{os.environ['TMP']}/*.png||". 
    </system>

    <input>
    You have the following files to chat with: {pdf_paths}
    You have the following CSVs to chat with: {csv_paths}
    Here is your question: <question>{query}</question> 
    </input>""".strip(), 'chat_history': chat_history})

    if self.extract_img_path(result['output']): 
      result['file_path'] = f"{self.extract_img_path(result['output'])}"
    
    return result

  def extract_img_path(self, text): 
    pattern = r'[\w/.\-]+/\w+.\w+'

    match = re.search(pattern, text)
    if match:
        file_path = match.group()
        return file_path
    else:
        return None 