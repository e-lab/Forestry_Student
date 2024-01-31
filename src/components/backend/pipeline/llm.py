from components.backend.tools.python_interpreter import PythonInterpreter
from components.backend.tools.arxiv_search import ArxivSearch
from components.backend.tools.calculator import Calculator
from components.backend.tools.web_search import WebSearch
from langchain_openai import ChatOpenAI

import os, re, json

class LLM: 
  def __init__(self, temperature=0.0001): 
    self.llm = ChatOpenAI(model_name='gpt-4', temperature=temperature)
  
  