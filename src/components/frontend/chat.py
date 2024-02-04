import streamlit as st
import os, re, json
import base64
import extra_streamlit_components as stx
from annotated_text import annotated_text
import datetime
from langchain_core.messages import AIMessage, HumanMessage


class Chat_UI:
    def __init__(self, pipeline, cookie_manger):
        self.pipeline = pipeline
        self.cookie_manger = cookie_manger

    def render(self):
        self.chat()

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

    def __call__(self):
        # Instantiates the chat history
        self.initiate_memory()
        self.load_memory()

        # Load's the text tab
        self.load_chatbox()

    def load_chatbox(self):
        user_input = st.text_input(
            label="*Got a question?*",
            help="Try to specify keywords and intent in your question!",
            key="text",
        )

        st.button(label="Send", key="send", on_click=self.handle_query)

        if st.button(label="Delete History", use_container_width=True, type="primary"):
            self.delete_messages()

    def load_memory(self):
        messages = self.get_messages()
        # print("memory loading: ", messages)
        if messages:
            for message in messages:
                role = message["role"]
                content = message["content"]

                with st.chat_message(role):
                    if type(content) == dict and role == "assistant":
                        st.json(content)

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

    def handle_query(self):
        text = st.session_state["text"]
        st.session_state["text"] = ""

        user_message = {"role": "user", "content": text}
        self.append(user_message)

        with st.chat_message("user"):
            st.markdown(text)

        with st.chat_message("assistant"):
            idx, tool = 0, None

            with st.spinner("Thinking..."):
                results = self.pipeline.run(
                    query=text, chat_history=self.format_history()
                )
            
            # print("Results: ", results)
            st.json({
                key: value for key, value in results.items() if key != "chat_history"
            })

        assistant_message = {
            "role": "assistant",
            "content": {
                key: value for key, value in results.items() if key != "chat_history"
            },
        }
        self.append(assistant_message)

        self.store_messages(user_message, assistant_message)

        # print("Messages: ", st.session_state["messages"])

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
        self.initiate_memory()
