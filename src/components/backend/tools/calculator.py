from langchain.chains import LLMMathChain
from langchain.tools import Tool
from pydantic import BaseModel, Field

class Calculator: 
  def __init__(self, llm): 
    self.llm = llm 

  def initialize(self): 
    return Tool.from_function(
        func=LLMMathChain.from_llm(llm=self.llm, verbose=True).run,
        name="Calculator",
        description="useful for when you need to answer questions about math. Pass in an equation as a string.",
    )