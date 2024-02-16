from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os 
import pandas as pd 
import uuid 
import streamlit as st 
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

@st.cache_data(ttl=1800, max_entries=70, show_spinner=False)
def load_docs(paths, _document_handler): 
  return RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=20).create_documents(_document_handler.load(paths))

class VectorStore: 
  def __init__(self): 
    self.embeddings_model = OpenAIEmbeddings(openai_api_key=st.session_state['api_key'])
    self.vectorstore = Chroma.from_documents(
      [Document(page_content='', metadata={'source': 'local'})],
      self.embeddings_model
    )
  
  def as_retriever(self, paths=None, document_handler=None):
    if paths: 
      return Chroma.from_documents(
        load_docs(paths, document_handler),
        self.embeddings_model
      ).as_retriever() 
    else: 
      return Chroma.from_documents(
          [Document(page_content='', metadata={'source': 'local'})], 
          self.embeddings_model
      ).as_retriever() 