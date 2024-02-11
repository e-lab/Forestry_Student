import streamlit as st
import os, re, json
import base64
import extra_streamlit_components as stx
from annotated_text import annotated_text
import datetime
from langchain_core.messages import AIMessage, HumanMessage
import glob 

class Chat_UI:
    def __init__(self, pipeline, cookie_manger):
        self.pipeline = pipeline
        self.cookie_manger = cookie_manger

    def initiate_memory(self):
        history = self.get_messages()

        if not history:
            st.session_state["messages"] = [
                {
                    "role": "assistant",
                    "content": "Hello! The name's euGenio. I'm here to help you with your pipeline. Ask me a question!",
                }
            ]
        else:
            st.session_state["messages"] = history

    def append(self, message: dict):
        st.session_state["messages"].append(message)

    def initialize(self):

        st.markdown("""
        <style>
               .block-container {
                    padding-top: 0.2rem;
                    padding-bottom: 2rem;
                }
        </style>
        """, unsafe_allow_html=True)

        # Instantiates the chat history
        if "query" not in st.session_state: 
            st.session_state.query = ""

        self.initiate_memory()
        self.load_memory()
        st.divider()
        self.load_chatbox()
        st.divider()
        
    def load_chatbox(self):
        col1, col2 = st.columns([5,1]) 
        with col1: 
            user_input = st.text_input(
                label='',
                placeholder="Enter your question here:",
                label_visibility='collapsed',
                help="Try to specify keywords and intent in your question!",
                key="user_input"
            )

        with col2: 
            submit_text = st.button(label="Send", use_container_width=True, type="primary", key="submit_text")

        rag  = st.multiselect('Documents', 
            [fil.replace(f"{os.environ['TMP']}", '') for fil in glob.glob(f"{os.environ['TMP']}/*.pdf")]+[fil.replace(f"{os.environ['TMP']}", '') for fil in glob.glob(f"{os.environ['TMP']}/*.txt")], 
            placeholder="Select Document(s) to chat with", key="rag")
        
        csv  = st.multiselect('Data Files', 
            [fil.replace(f"{os.environ['TMP']}", '') for fil in glob.glob(f"{os.environ['TMP']}/*.csv")], 
            placeholder="Select CSV(s) to chat with", key="csv")
                

        if st.button(label="Delete History", use_container_width=True):
            self.delete_messages()

        if submit_text and user_input: 
            self.handle_query(user_input, csv, rag)
            user_input = ""

    def load_memory(self):
        messages = st.session_state["messages"] 
        # print("memory loading: ", messages)
        if messages:
            for message in messages:
                role = message["role"]
                content = message["content"]

                with st.chat_message(role):
                    if type(content) == dict and role == "assistant":
                        st.json({key: value for key, value in content.items() if key != "file_path"})
                        if 'file_path' in content and content['file_path']:
                            st.image(content["file_path"])

                    else:
                        st.markdown(content)

    def format_history(self):
        messages = st.session_state["messages"]

        if messages:
            formatted = []
            for message in messages[1:]:
                if message["role"] == "user":
                    formatted.append(HumanMessage(content=str(message["content"])))
                else:
                    formatted.append(AIMessage(content=str(message["content"])))
            return formatted
        else:
            return []

    def handle_query(self, text, csv, rag):
        user_message = {"role": "user", "content": text}
        self.append(user_message)

        with st.chat_message("user"):
            st.markdown(text)

        with st.chat_message("assistant"):
            idx, tool = 0, None

            with st.spinner("Thinking..."):
                results = self.pipeline.run_normal(
                    query=text, chat_history=self.format_history(), 
                    csv_paths=[f"{os.environ['TMP']}"+c for c in csv], 
                    pdf_paths=[f"{os.environ['TMP']}"+r for r in rag]
                )

                print(results)


            with st.expander('Thought Process:', expanded=False): 
                st.json({
                    'input': text,
                    'output': results['output'],
                })


            if 'file_path' in results and results["file_path"]:
                st.image(results["file_path"])
            else: 
                st.markdown(f"`{results['output']}`")


        assistant_message = {
            "role": "assistant",
            "content": {
                    'input': text,
                    'output': results['output'],
                    'file_path': results['file_path'] if "file_path" in results else ''
            },
        }

        self.append(assistant_message)

        self.store_messages(user_message, assistant_message)

    def store_messages(self, user_message, assistant_message):
        past = st.session_state["messages"]
        # print("Past", past)

        # print("Entered if past")
        self.cookie_manger.set("messages", past)
        # print("Messaged set")

    def get_messages(self):
        return self.cookie_manger.get(cookie="messages")

    def delete_messages(self):
        self.cookie_manger.delete(cookie="messages")
