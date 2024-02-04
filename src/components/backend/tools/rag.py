from langchain.chains import RetrievalQA
from langchain.tools import Tool

class RAG: 
  def __init__(self, llm, vectorstore): 
    self.llm = llm 
    self.vectorstore = vectorstore

  def run(self, query):
    retrieval_qa = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff",
        retriever=self.vectorstore.as_retriever())
 
    answer = retrieval_qa.invoke(query)
    
    del retrieval_qa

    return answer
