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
        self.cookie_manager = stx.CookieManager()

        if 'api_key' not in st.session_state: 
            st.session_state['api_key'] = None
            self.pipeline = None
        if 'messages' not in st.session_state: 
            st.session_state['messages'] = [] 
        if 'user_id' not in st.session_state: 
            st.session_state['user_id'] =  str(uuid.uuid4())    

        _ = self.cookie_manager.get_all()
        self.sidebar = Sidebar(None)
        self.chat = Chat_UI(None, self.cookie_manager)

        
    def render(self): 
        st.markdown("""
        <style>
               .block-container {
                    padding-top: 0.5rem;
                    padding-bottom: 0.2rem;
                }
        </style>
        """, unsafe_allow_html=True)

        status = self.sidebar() 

        empty = st.empty() 

        print("I'm being reloaded now")
        print("API Key: ", st.session_state['api_key'])

        if not status: 
            with empty: 
                with st.chat_message("assistant"):
                    st.markdown("Please enter your `OPENAI_API_KEY` to begin!")
        else: 
            empty.empty()
            self.pipeline = Pipeline()
            self.sidebar.pipeline = self.pipeline
            self.chat.pipeline = self.pipeline

            st.session_state['documents'] = len(self.pipeline.document_handler)

            self.sidebar._upload_widget()
            self.sidebar._delete_widget()
            self.chat.initialize()


def main(): 
    UI().render()

if __name__ == "__main__": 
    main() 