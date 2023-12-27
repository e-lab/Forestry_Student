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
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chains import LLMMathChain
from langchain.utilities import ArxivAPIWrapper
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


from langchain.llms import Ollama
from langchain.chat_models import ChatOllama
from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    question: str = Field()

class DataframeInput(BaseModel):
    query: str
    csv_path: str
    chat_history: list = [] 


class ChatBot:
    def __init__(self, embedding_size=1536,temperature=0.0001, max_iterations=3):
        # Set up API keys and environment variables
        os.environ["OPENAI_API_KEY"] = "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"
        os.environ['PINECONE_API_KEY'] = "204755b4-f7d8-4aa4-b16b-764e66796cc3"
        openai.api_key = os.environ['OPENAI_API_KEY']
        os.environ["GOOGLE_API_KEY"] = "AIzaSyDKxAadUfBZ9oAMDlRjRe0jlp3N0oZKqvg"
        os.environ["GOOGLE_CSE_ID"] = "57d010b1a25ce48c0"
        
        # Initialize database
        self.init_database()

        #Initalize backup_check
        with open("backup_check.pkl", 'rb') as file:
            data = pickle.load(file)
        #print(data['embedding'])
        self.backup_check = data['embedding']
        """
        self.llm = Ollama(model="llama2:13b")
        
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
        self.llm = ChatOpenAI(model_name='gpt-4', temperature=temperature)
        self.embeddings_model = OpenAIEmbeddings()
        self.embedding_size = embedding_size



        self.index = faiss.IndexFlatL2(self.embedding_size)
        self.vectorstore_public = FAISS(self.embeddings_model.embed_query, self.index, InMemoryDocstore({}), {})
        self.search = GoogleSearchAPIWrapper()
        self.web_retriever = WebResearchRetriever.from_llm(
            vectorstore=self.vectorstore_public,
            llm=self.llm, 
            search=self.search, 
            num_search_results=3
        )


        self.web_QA = RetrievalQAWithSourcesChain.from_chain_type(self.llm, retriever=self.web_retriever)
        llm_math_chain = LLMMathChain.from_llm(llm=self.llm, verbose=True)
        arxiv = ArxivAPIWrapper()
        
        #Define Tools 
        
        self.tools = [ Tool(
            name="in_context_qa",
            func=self.in_context_qa,
            description="The in_context_qa function enhances an LLM agent's ability to respond to user queries within a specific database context. This tool should ALWAYS be called before trying to search the web, It takes a user question (query), loads a specified database (database_name), and calculates similarity to determine context relevance. If similarity is high, the LLM generates a response combining the user query and database content. The function outputs a formatted response and updates the chat history for ongoing context preservation."
            ),
            Tool.from_function(
                func=self.web_QA,
                name="web_QA",
                description="web_QA is a web searching tool for the LLM agent, triggered when the similarity score from in-context QA is too low. It dynamically integrates the LLM and a web retriever to broaden knowledge through targeted web searches, enhancing the agent's responsiveness and adaptability to diverse user queries",
            ),
            Tool.from_function(
                func=llm_math_chain.run,
                name="Calculator",
                description="useful for when you need to answer questions about math",
                args_schema=CalculatorInput,
                # coroutine= ... <- you can specify an async method if desired as well
            ),
            Tool.from_function(
                func=arxiv.run,
                name="arxiv",
                description="useful for when you need to answer research based questions or find scientific documents or papers",
                args_schema=CalculatorInput,
                # coroutine= ... <- you can specify an async method if desired as well
            ),
            
            Tool.from_function(
                func=self.parsing_dataframe_question,
                name="parsing_dataframe_question",
                description="""This notebook shows how to use agents to interact with a Pandas DataFrame. It is mostly optimized for question answering.
                            The input to this tool should be a comma separated list of strings of length two, representing the two inputs first the user question and second the csv file path""",
                
                # coroutine= ... <- you can specify an async method if desired as well
            )
        ]
        
        self.agent = initialize_agent(self.tools, 
                                        self.llm, 
                                        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                                        verbose=True, 
                                        handle_parsing_errors=True, 
                                        max_iterations=max_iterations)
        #print(f"Agent Prompt \n\t{self.agent.agent.llm_chain.prompt.template}\n\n")




        

    def add_embeddings(self, file_contents, file_name, class_name, topic_name, database_name = 'courses.db'):
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

        
    def init_database(self, database_name = 'courses.db'):
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

    

    def load_database(self, database_name = 'courses.db'):
        self.conn = sqlite3.connect(database_name)
        self.cursor = self.conn.cursor()

    
    
    def in_context_qa(self, query, database_name='courses.db', chat_history=[]):
        self.load_database(database_name)
        # Process the user's question and get the most similar embedding from the database
        #print(query)
        vector_content, _, file_name, similarity_score = np.array(get_most_similar_embbeding_for_question(self.cursor,query))
        #print(similarity_score)

        # Check similarity threshold and respond accordingly
        if similarity_score < 0.7:
            return "We appear to lack a strong enough similarity between our database and your question...\nWe are searching the web for your answer!", chat_history
        else:
            # If similarity is high, use the local database
            llm_response = self.llm.predict(f"{vector_content} {query}")
            
            """NEED BETTER WAY TO AVOID OUTPUTTING CONTENT IF LLM DOSEN"T NEED IT"""
            vector_content = vector_content.replace("\n", "").replace("\t", "")
            #print(f"\n\n\n {llm_response}")
            #print(f"\n\nLLM Response:\n\n{llm_response}\n")
            #print(f"Source: \n\n{vector_content}______________________\n\n")
            message = f"{llm_response}\n\n Cotent:\n{vector_content}\n\nFile Name:\n{file_name}"
            chat_history.append({'user': False, 'message': message})

        return message, chat_history

    def parsing_dataframe_question(self, string):
        print("parsing_dataframe_question TEST")
        print(string)
        if type(string) is not str:
            string = ', '.join(string)
        query, csv_path = string.split(",")
        
        return self.process_dataframe_question(str(query), str(csv_path))
    
    def process_dataframe_question(self, query, csv_path, chat_history=[]):
        df = pd.read_csv(csv_path)
        df.attrs['filename']=csv_path
        #print(df.attrs['filename'])
        agent = create_pandas_dataframe_agent(self.llm, df, verbose=True, agent="chat-zero-shot-react-description")
        response = agent.run(f"{query}")
        #print(response)
        return response


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
