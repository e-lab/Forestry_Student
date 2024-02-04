from langchain.chains import RetrievalQA
from langchain.tools import Tool

from langchain.tools.retriever import create_retriever_tool

class RAG:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def initialize(self):
        return Tool.from_function(
            func=self.vectorstore.get,
            name="VectorDB",
            description="Always begin with this tool. If there is no input documents, then move on to the web_QA. This tool returns documents from the vectorstore based on a query. Pass in a query and get back a list of documents.",
        )

