from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.utilities import PythonREPL
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate

class PythonInterpreter: 
  def __init__(self, llm):
    self.llm = llm 

  def _sanitize_output(self, text: str):
      _, after = text.split("```python")
      return after.split("```")[0]

  def python_interpreter(self, query):
      template = """Write some python code to solve the user's problem. 

      Return only python code in Markdown format, e.g.:

      ```python
      ....
      ```"""
      prompt = ChatPromptTemplate.from_messages([("system", template), ("human", "{input}")])
      chain = prompt | self.llm | StrOutputParser() | self._sanitize_output | PythonREPL().run
      output = chain.invoke({"input": query})
      print("Python interpreter")
      print(output)
      return output
      
  def initialize(self): 
    return Tool.from_function(
      func=self.python_interpreter,
      name="python_interpreter",
      description="""The Python Code Generator Tool is a sophisticated utility designed to craft Python code solutions for a wide array of questions. When provided with a question, this tool leverages advanced algorithms to generate concise and efficient Python code snippets as answers.

                      Usage Instructions:

                      Pose a question requiring a Python code solution.
                      If existing tools are deemed insufficient for the task, instruct the Assistant to utilize the Python Code Generator Tool.
                      Expect a response in the form of a Markdown-formatted Python code block, enclosed within triple backticks.""",
    )