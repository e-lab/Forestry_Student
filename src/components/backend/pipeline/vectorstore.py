from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os 
import pandas as pd 
import uuid 
import streamlit as st 
from langchain.docstore.document import Document
from langchain_experimental.text_splitter import SemanticChunker

class VectorStore: 
  def __init__(self): 
    self.embeddings_model = OpenAIEmbeddings(openai_api_key=st.session_state['api_key'])
    self.vectorstore = Chroma.from_documents(
      [Document(page_content='', metadata={'source': 'local'})],
      self.embeddings_model
    )
  
  def as_retriever(self, paths=None, document_handler=None):
    if paths: 
      print(document_handler.load(paths))
      text_splitter = SemanticChunker(self.embeddings_model)
      return Chroma.from_documents(
        text_splitter.create_documents(document_handler.load(paths)),
        self.embeddings_model
      ).as_retriever() 
    else: 
      return Chroma.from_documents(
          [Document(page_content='', metadata={'source': 'local'})], 
          self.embeddings_model
      ).as_retriever() 