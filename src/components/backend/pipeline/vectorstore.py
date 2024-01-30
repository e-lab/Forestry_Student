import chromadb 
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

class VectorStore: 
  def __init__(self): 
    self.chroma_client = chromadb.Client()
    _ = self.chroma_client.create_collection(name="user")
    self.embeddings_model = OpenAIEmbeddings()

    self.vectorstore = Chroma(
        client=self.chroma_client,
        collection_name="user",
        embedding_function=self.embeddings_model,
    )
