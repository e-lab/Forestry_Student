from langchain_community.utilities import ArxivAPIWrapper
from langchain.tools import Tool
from pydantic import BaseModel, Field


class ArxivSearch: 
  def __init__(self): 
    self.arxiv = ArxivAPIWrapper()

  def initialize(self):
    return Tool.from_function(
        func=self.arxiv.run,
        name="arxiv",
        description="Useful for when you need to answer research based questions or find scientific documents or papers. Use this as the last step for topics you do not understand.",
    )