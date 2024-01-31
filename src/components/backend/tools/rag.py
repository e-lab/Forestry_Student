from langchain.chains import RetrievalQA
from langchain.tools import Tool

class RAG: 
  def __init__(self, llm, vectorstore): 
    self.retrieval_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
        retriever=vectorstore.as_retriever())

  def initialize(self): 
    return Tool(
        name = "Document Store",
        func = self.retrieval_qa.run,
        description = "Use it to lookup information from document store. \
                      Always used as first tool"
    )
