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
import extra_streamlit_components as stx

# "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"

st.set_page_config(layout='wide')

class UI: 
    def __init__(self): 
        self.cookie_manager = stx.CookieManager()

        if 'api_key' not in st.session_state: 
            st.session_state['api_key'] = None 
        if 'messages' not in st.session_state: 
            st.session_state['messages'] = [] 


        cookies = self.cookie_manager.get_all()
        st.session_state['documents'] = False
        st.session_state['user_id'] = str(uuid.uuid4())

        self.sidebar = Sidebar(None)
        self.chat = Chat_UI(None, self.cookie_manager)

        
    def render(self): 
      status = self.sidebar() 
      self.chat() 
      if status: 
        self.pipeline = Pipeline()
        self.sidebar.pipeline = self.pipeline
        self.chat.pipeline = self.pipeline
        self.chat.load_chatbox()

def main(): 
    UI().render()

if __name__ == "__main__": 
    main() 