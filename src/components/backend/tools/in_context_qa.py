from langchain.chains import RetrievalQA
from langchain.tools import Tool
from langchain.tools.retriever import create_retriever_tool

class InContextQA:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def initialize(self, path, document_handler):
        return create_retriever_tool(
            self.vectorstore.as_retriever(path, document_handler),
            "in_context_qa",
            "Searches for documents, and returns the most relevant ones. Lower distance score means higher relevance. Use this to answer questions with unfamiliar terms and people.",
        )

