import chromadb 
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import chromadb.utils.embedding_functions as embedding_functions
import os 
import pandas as pd 
import uuid 

class VectorStore: 
  def __init__(self): 
    self.chroma_client = chromadb.Client()
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ["OPENAI_API_KEY"],
                model_name="text-embedding-ada-002"
            )
    self.collection = self.chroma_client.get_or_create_collection(name="user", embedding_function=openai_ef)
    self.embeddings_model = OpenAIEmbeddings()

    self.vectorstore = Chroma(
        client=self.chroma_client,
        collection_name="user",
        embedding_function=self.embeddings_model,
    )
  
  def as_retriever(self):
    return self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3}) 
  
  def add(self, text_blocks):
    df = pd.DataFrame(text_blocks, columns=['id', 'page_num', 'xmin', 'ymin', 'xmax', 'ymax', 'text'])

    assert len(set(df['id'])) == 1

    uuids = [str(uuid.uuid4()) for _ in range(len(df))]

    self.collection.add(
        documents=df['text'].tolist(),
        metadatas=df[['id', 'page_num', 'xmin', 'ymin', 'xmax', 'ymax', 'text']].to_dict(orient='records'),
        ids=uuids
      )
    
    del df 
    
    return 1 

