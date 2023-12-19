from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from openai.error import OpenAIError
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.vectorstores import FAISS 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore import InMemoryDocstore  
from langchain.utilities import GoogleSearchAPIWrapper
from sklearn.metrics.pairwise import cosine_similarity
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

import faiss
import openai
import sqlite3
import pickle
import json
import time
import os
import numpy as np
import pandas as pd
from openai.error import OpenAIError
from utils import get_most_similar_embbeding_for_question, get_embedding, insert_data_into_database, split_to_chunks

import textwrap

from langchain.llms import Ollama
from langchain.chat_models import ChatOllama

class ChatBot:
    def __init__(self):
        # Set up API keys and environment variables
        os.environ["OPENAI_API_KEY"] = "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"
        os.environ['PINECONE_API_KEY'] = "204755b4-f7d8-4aa4-b16b-764e66796cc3"
        openai.api_key = os.environ['OPENAI_API_KEY']
        os.environ["GOOGLE_API_KEY"] = "AIzaSyDKxAadUfBZ9oAMDlRjRe0jlp3N0oZKqvg"
        os.environ["GOOGLE_CSE_ID"] = "57d010b1a25ce48c0"
        
        # Initialize database
        self.init_database()

        self.llm = Ollama(model="llama2:13b")
        """
        agent = initialize_agent(
        agent="chat-conversational-react-description",
        tools=tools,
        llm=llm,
        verbose=True,
        early_stopping_method="generate",
        memory=memory,
        # handle_parsing_errors=True,
        #agent_kwargs={\"output_parser\": parser}
    )
    """
        # Initialize models and retrievers
        self.llm = ChatOpenAI(model_name='gpt-4', temperature=0)
        self.embeddings_model = OpenAIEmbeddings()
        self.embedding_size = 1536
        self.index = faiss.IndexFlatL2(self.embedding_size)
        self.vectorstore_public = FAISS(self.embeddings_model.embed_query, self.index, InMemoryDocstore({}), {})
        self.search = GoogleSearchAPIWrapper()
        self.web_retriever = WebResearchRetriever.from_llm(
            vectorstore=self.vectorstore_public,
            llm=self.llm, 
            search=self.search, 
            num_search_results=3
        )
        self.qa_chain = RetrievalQAWithSourcesChain.from_chain_type(self.llm, retriever=self.web_retriever)
        """NEED A BETTER WAY TO EMBEDD THIS SO THAT WE DONT HAVE TO PAY EVERY TIME A NEW INSTANCE RUNS"""
        #self.backup_check = get_embedding("I'm sorry, but I couldn't find any information about")
        #data = {'embedding': self.backup_check }
        #with open("backup_check.pkl", 'wb') as file:
        #    pickle.dump(data, file)

        with open("backup_check.pkl", 'rb') as file:
            data = pickle.load(file)
        #print(data['embedding'])
        self.backup_check = data['embedding']

    def add_embeddings(self, file_contents, file_name, class_name, topic_name, database_name = 'trees.db'):
        database_exists = os.path.exists(database_name)
        # If the database doesn't exist, create the table
        if not database_exists:
            self.init_database(database_name)
        self.conn = sqlite3.connect(database_name)
        self.cursor = self.conn.cursor()

        
        texts = split_to_chunks(file_contents)
        for i, t in enumerate(texts):
            insert_data_into_database(self.cursor, t, class_name, topic_name, file_name)
        self.conn.commit()
        self.conn.close()

        
    def init_database(self, database_name = 'trees.db'):
        # Check if the database exists
        
        database_exists = os.path.exists(database_name)

        # If the database doesn't exist, create the table
        if not database_exists:
            print("MAKING DB")
            # Connect to SQLite database or create it if it doesn't exist
            self.conn = sqlite3.connect(database_name)
            self.cursor = self.conn.cursor()
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT,
                    embedding BLOB,
                    class TEXT,
                    topic TEXT,
                    file_name TEXT
                )
            ''')
            self.conn.commit()
            self.conn.close()

    

    def load_database(self, database_name = 'trees.db'):
        self.conn = sqlite3.connect(database_name)
        self.cursor = self.conn.cursor()

    def format_string(self, input_string, n):
        formatted_string = textwrap.wrap(input_string, width=n)
        return '\n'.join(formatted_string)

    def process_question(self, query, csv_path, chat_history=[]):
        df = pd.read_csv(csv_path)
        df.attrs['filename']=csv_path
        print(df.attrs['filename'])
        agent = create_pandas_dataframe_agent(self.llm, df, verbose=True, agent="chat-zero-shot-react-description")
        response = agent.run(f"{query}")
        response = self.format_string(response, n=60)
        #print("_____")
        print(response)



        chat_history.append({'user': False, 'message': f"{response}"})
        return chat_history


# Example usage
if __name__ == "__main__":
    chat_bot = ChatBot()
    csv_path = "/Users/viktorciroski/Documents/Github/Forestry_Student/TreeHugger_Exam_ans.csv"
    while True:
        print("______________________\n")
        user_query = input("What is your question? Enter 'Stop' to exit\t")
        if user_query.lower() == "stop":
            break
        chat_bot.process_question(user_query, csv_path, chat_history=[])
        #chat_bot.process_question(user_query)
