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

import faiss
import openai
import sqlite3
import json
import time
import os
import numpy as np
import time
from openai.error import OpenAIError
import os
import json

from utils import get_most_similar_embbeding_for_question, get_embedding, insert_data_into_database, split_to_chunks


os.environ["OPENAI_API_KEY"] = "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" # Get it at https://beta.openai.com/account/api-keys
os.environ['PINECONE_API_KEY'] = "204755b4-f7d8-4aa4-b16b-764e66796cc3"
os.environ["GOOGLE_API_KEY"] = "AIzaSyDKxAadUfBZ9oAMDlRjRe0jlp3N0oZKqvg" # Get it at https://console.cloud.google.com/apis/api/customsearch.googleapis.com/credentials
os.environ["GOOGLE_CSE_ID"] = "57d010b1a25ce48c0" # Get it at https://programmablesearchengine.google.com/
search = GoogleSearchAPIWrapper()   
database_name='trees.db'

database_exists = os.path.exists(database_name)
# If the database doesn't exist, create the table
if not database_exists:
    # Connect to SQLite database or create it if it doesn't exist
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT,
            embedding BLOB,
            class TEXT,
            topic TEXT
        )
    ''')
    conn.commit()
    conn.close()


#Get Data from User 
file_path = 'indiana_trees.txt'
with open(file_path, 'r') as file:
    file_contents = file.read()
#print(file_contents)

class_name = input("Enter Class Name\t")
topic_name = input("Enter Topic Name\t")

texts = split_to_chunks(file_contents)
#print(texts)


# Connect to SQLite database or create it if it doesn't exist
conn = sqlite3.connect(database_name)
cursor = conn.cursor()

# Example usage in a loop
for i, t in enumerate(texts):
    insert_data_into_database(cursor, t, class_name, topic_name)


llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
# Initialize 
embeddings_model = OpenAIEmbeddings()  
embedding_size = 1536  
index = faiss.IndexFlatL2(embedding_size)  
vectorstore_public = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
web_retriever = WebResearchRetriever.from_llm(
    vectorstore=vectorstore_public,
    llm=llm, 
    search=search, 
    num_search_results=3
)
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=web_retriever)
"""Figure out how to make custom retrieve for converstation 
retriever = vectordb.as_retriever()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages= True)
chain = ConversationalRetrievalChain.from_llm(llm, retriever= retriever, memory= memory)
"""

backup_check = get_embedding("I'm sorry, but I couldn't find any information about")
while True:
    print("______________________\n")
    query = input("What is your question? Enter Stop to exit\t")
    if query.lower() == "stop":
        break
    vector_content, vector_chunck_embedding, similarity_score = np.array(get_most_similar_embbeding_for_question(cursor, query))
    #print(vector_chunck_embedding[0])
    print(similarity_score)
    #print(vector_content)
    """Here we can use the similarty threshold to serach the web"""
    if similarity_score < 0.6:
        print("\n\nWe appear to lack a strong enough similarity between our database and your question...\nWe are searching the web for your answer!")
        llm_response = qa_chain({"question": query})
        print(f"\n\nLLM Response:\n\n{llm_response['answer']}\n")
        print(f"Sources:\n\n  {llm_response['sources']}\n______________________\n\n")

    else:
        
        llm_response = llm.predict(f"{vector_content} {query}")
        if cosine_similarity(np.array(get_embedding(llm_response) ).reshape(1, -1), np.array(backup_check).reshape(1, -1)) >0.8:
            print("\n\nWe appear to lack a strong enough similarity between our database and your question...\nWe are searching the web for your answer!")
            llm_response = qa_chain({"question": query})
            print(f"\n\nLLM Response:\n\n{llm_response['answer']}\n")
            print(f"Sources:\n\n  {llm_response['sources']}\n______________________\n\n")

            print(f"\n\n\n {llm_response}")
        else:
            print(f"\n\nLLM Response:\n\n{llm_response}\n")
            print(f"Source: \n\n{vector_content}______________________\n\n")





conn.commit()
conn.close()