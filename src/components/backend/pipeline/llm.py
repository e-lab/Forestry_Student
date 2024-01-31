from langchain_openai import ChatOpenAI

import os, re, json

class LLM: 
  def __init__(self, temperature=0.0001): 
    self.llm = ChatOpenAI(model_name='gpt-4', temperature=temperature)
  
  