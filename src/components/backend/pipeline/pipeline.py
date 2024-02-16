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

@st.cache_data(ttl=3600, max_entries=10, show_spinner=False) 
def load_hub(link): 
  return hub.pull(link)

@st.cache_resource(ttl=1800, max_entries=100, show_spinner=False) 
def load_agent(pdf_paths, csv_paths, max_iterations, _document_handler): 
    document_handler = _document_handler
    llm = LLM()
    vectorstore = VectorStore() 

    tools = [
      PythonInterpreter(llm=llm.llm).initialize(),
      ArxivSearch().initialize(),
      Calculator(llm=llm.llm).initialize(),
      WebSearch(llm=llm.llm, vectorstore_public=vectorstore.vectorstore).initialize(),
    ]
    
    if pdf_paths: 
      tools += [InContextQA(vectorstore=vectorstore).initialize(pdf_paths, document_handler)]
    if csv_paths:
      tools += [CSVAgent(llm.llm, csv_paths).initialize()]

    tools_agent = create_openai_tools_agent(llm.llm, 
            tools,
            load_hub("hwchase17/openai-functions-agent"),
          )
    
    tools_agent = AgentExecutor(agent=tools_agent, tools=tools, verbose=True, handle_parsing_errors=True, 
        max_iterations=max_iterations)

    return tools_agent

class Pipeline: 
  def __init__(self, max_iterations=5): 
    self.document_handler = Document_Handler() 
    self.max_iterations = max_iterations

  def run_normal(self, query, chat_history, csv_paths=[], pdf_paths=[]):
    print(pdf_paths, csv_paths)

    tools_agent = load_agent(pdf_paths, csv_paths, self.max_iterations, self.document_handler)

    prompt = """ 
    <system> 
    """

    if pdf_paths: 
      prompt += """ 
    IF YOU ARE ASKED ABOUT PEOPLE or SCIENTIFIC TERMS: START WITH in_context_qa.
    FOR UNKNOWN NOUNS, NAMES AND OBJECTS: START WITH in_context_qa, THEN arxiv_search.
    """
    else: 
      prompt += """ 
    IF YOU ARE ASKED ABOUT PEOPLE or SCIENTIFIC TERMS: START WITH web_search.
    FOR UNKNOWN NOUNS, NAMES AND OBJECTS: START WITH web_search and then arxiv_search.
    """

    if csv_paths: 
      prompt += """ 
    FOR QUESTIONS ABOUT DATA: START WITH csv_agent.
    IF YOU ARE ASKED TO VISUALIZE DATA: START WITH csv_agent, ASK IT TO MAKE A DF QUERYING COMMAND, PASS IT TO python_interpreter TO USE THE QUERY TO SAVE A VISUALIZATION. In this case, return ONLY "||{os.environ['TMP']}/*.png||". The path will be parsed, ensure it is the only thing returned. 
    """
    
    
    prompt += """ 
    </system>

    <input>
    """

    if pdf_paths: 
      prompt += f""" 
    You have the following files to chat with: {pdf_paths}
    """

    if csv_paths: 
      prompt += f""" 
    You have the following CSVs to chat with: {csv_paths}
    """

    prompt += f""" 
    Here is your question: <question>{query}</question> 
    
    </input>
    """

    print('Prompt: \n', prompt)

    result = tools_agent.invoke({'input': prompt.strip(), 'chat_history': chat_history})

    if self.extract_img_path(result['output']): 
      result['file_path'] = f"{self.extract_img_path(result['output'])}"
    
    return result

  def extract_img_path(self, text): 
    pattern = r'[\w/.\-]+/\w+.\w+'

    match = re.search(pattern, text)
    if match:
        file_path = match.group()
        if file_path.endswith('.png'): 
          return file_path
        else: 
          return None 
    else:
        return None 