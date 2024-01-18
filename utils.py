from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


from openai import OpenAI



import sqlite3
import json
import time
import os

import time
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import sqlite3
import json

client =  OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def get_most_similar_embbeding_for_question(cursor, question):
    #Probably need to send in cursor
    """
    Retrieve the most similar embedding from the database for the given question.

    Parameters:
    - question (str): The input question for which the similarity is calculated.

    Returns:
    - embeddings (list or None): The embeddings associated with the most similar question, or None if no result.
    """

    # Assuming you have a database connection object named 'cursor'
    cursor.execute("SELECT content, embedding, file_name FROM documents")
    result = cursor.fetchall()

    q_embbed = get_embedding(question)
    start_time = time.time()

    result_dict = {"Content": [], 'Embeddings': [], 'file_name': [], 'Similarity Score': []}

    #print("SIMILARITY SCORE")
    if result:
        for index in range(len(result)):
            content = result[index][0]
            r = result[index][1]
            f = result[index][2]
        
            r = json.loads(r)
            similarity_score = cosine_similarity(np.array(q_embbed).reshape(1, -1), np.array(r).reshape(1, -1))
            result_dict['Embeddings'].append([r])
            result_dict['Similarity Score'].append(similarity_score[0][0])
            result_dict['Content'].append(content)
            result_dict['file_name'].append(f)
 
        result_df = pd.DataFrame(result_dict)
        max_similarity_row = result_df.loc[result_df['Similarity Score'].idxmax()]

        end_time = time.time()
        runtime = end_time - start_time
        #print("Runtime:", runtime, "seconds")
        print(max_similarity_row)
        return max_similarity_row
            

    return None, None, None


def get_embedding(text_to_embed, max_retries=3, retry_delay=15):
    """
    Retrieves the embedding for a given text using OpenAI's text-embedding-ada-002 model.

    Parameters:
    - text_to_embed (str): The input text for which the embedding is to be generated.
    - max_retries (int, optional): The maximum number of retries in case of a ServiceUnavailable error. Defaults to 3.
    - retry_delay (int, optional): The delay in seconds between retry attempts. Defaults to 15.

    Returns:
    - list of floats: The embedding representation of the input text as a list of floats.
    """
    retries = 0
    while retries < max_retries:
        try:
            # Embed a line of text
            response = client.embeddings.create(model="text-embedding-ada-002",
            input=[text_to_embed])
            # Extract the AI output embedding as a list of floats
            embedding = response.data[0].embedding
            return embedding
        except Exception as e:
            if "ServiceUnavailable" in str(e) and retries < max_retries - 1:
                print(f"ServiceUnavailable error. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retries += 1
            else:
                raise

def insert_data_into_database(cursor, t, class_name, topic_name, file_name):
    """
    Inserts data into the database, handling cases where 't' may not have 'page_content' attribute.

    Parameters:
    - cursor (sqlite3.Cursor): The SQLite database cursor.
    - t: The object containing the text data.
    - class_name (str): The class to be inserted into the database.
    - topic_name (str): The topic to be inserted into the database.

    Returns:
    None
    """
    if hasattr(t, 'page_content'):
        content = t.page_content
    else:
        # Treat 't' like a string if it doesn't have 'page_content' attribute
        content = str(t)

    embedding_str = json.dumps(get_embedding(content))
    cursor.execute('''
        INSERT INTO documents (content, embedding, class, topic, file_name)
        VALUES (?, ?, ?, ?, ?)
    ''', (content, embedding_str, class_name, topic_name, file_name))

    



def split_to_chunks(data, chunk_size=1000, chunk_overlap=100):
    """
    Splits a given data string into chunks of specified size with an optional overlap.

    Parameters:
    - data (str): The input string to be split into chunks.
    - chunk_size (int, optional): The desired size of each chunk. Defaults to 1000.
    - chunk_overlap (int, optional): The number of overlapping characters between adjacent chunks. Defaults to 100.

    Returns:
    - list of str: A list containing the resulting chunks.
    """

    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size - chunk_overlap)]
    return chunks

