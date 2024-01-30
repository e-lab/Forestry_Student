# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# import sqlite3

import streamlit as st 
from components.frontend.chat import Chat_UI
from components.frontend.sidebar import Sidebar
from components.backend.pipeline.pipeline import Pipeline
import os 
import uuid 


st.set_page_config(layout='wide')


@st.cache_resource
def initalize(): 
    pipeline = Pipeline()
    return pipeline, Sidebar(pipeline), Chat_UI(pipeline)

class UI: 
    def __init__(self): 
        self._pipeline, self.sidebar, self.chat = initalize()
        st.session_state['documents'] = [0] 
        st.session_state['user_id'] = str(uuid.uuid4())
        st.session_state['api_key'] = "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"
        
        if 'messages' not in st.session_state: 
            st.session_state['messages'] = [] 
    
    def render(self): 
      self.sidebar() 
      self.chat() 
      
def main(): 
    UI().render()

if __name__ == "__main__": 
    main() 