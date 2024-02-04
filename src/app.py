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


st.set_page_config(layout='wide')

class UI: 
    def __init__(self): 
        self.pipeline = Pipeline()
        self.sidebar = Sidebar(self.pipeline)
        self.cookie_manager = stx.CookieManager()
        cookies = self.cookie_manager.get_all()
        print(cookies)
        self.chat = Chat_UI(self.pipeline, self.cookie_manager)
        st.session_state['documents'] = False
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