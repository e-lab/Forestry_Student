from langchain_openai import ChatOpenAI
import streamlit as st
import os, re, json

class LLM: 
  def __init__(self, temperature=0): 
    self.llm = ChatOpenAI(model_name='gpt-4', temperature=temperature, openai_api_key="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")
  
  