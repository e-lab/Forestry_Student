from langchain.agents.agent_types import AgentType
from langchain_experimental.agents import create_csv_agent
from langchain.tools import Tool

class CSVAgent: 
  def __init__(self, llm, csv_file_paths):
    if len(csv_file_paths) == 1:
      csv_file_paths = csv_file_paths[0]
    
    print(csv_file_paths)
    self.agent = create_csv_agent(
        llm,
        csv_file_paths,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        return_intermediate_steps=True
    )
  
  def invoke(self, query): 
    """Pass in a query regarding the dataframes in question, and this function will return information"""
    answer = self.agent(query)
    if answer["intermediate_steps"]:
      action = answer["intermediate_steps"][-1][0].tool_input["query"]
      print(answer, action)
      return (answer, action)
    return answer

  def initialize(self):
    return Tool.from_function( 
      func=self.invoke, 
      name="csv_agent",
      description="""The CSV Agent runs python code to answer your question about it's dataframes. 
      Pass in a query in the form a string and this function will return the information you want."""
    )