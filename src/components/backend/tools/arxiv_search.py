from langchain.utilities import ArxivAPIWrapper
from langchain.tools import Tool
from pydantic import BaseModel, Field


class ArxivSearch: 
  def __init__(self): 
    self.arxiv = ArxivAPIWrapper()

  def initialize(self):
    return Tool.from_function(
        func=self.arxiv.run,
        name="arxiv",
        description="useful for when you need to answer research based questions or find scientific documents or papers",
    )