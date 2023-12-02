from langchain.agents import AgentType, initialize_agent

from langchain.chat_models import ChatOpenAI

from langchain.agents import initialize_agent
from langchain.agents import AgentType, Tool
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from langchain.schema.messages import HumanMessage
from langchain.prompts.chat import MessagesPlaceholder

from typing import Tuple, Dict

import os 

from custom_tools import Image_to_Text

os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"
os.environ["OPENAI_API_KEY"] = "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" # Get it at https://beta.openai.com/account/api-keys


class ConfigAgent():
    """
    Contains the configuration of the LLM.
    """
    def __init__(self):
        self.model_name = 'gpt-3.5-turbo-16k-0613'

    def config(self, temperature=0):
        llm = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], 
                        temperature=temperature, 
                        model=self.model_name)
        return llm

    def setup_memory(self, max_history=5) -> Tuple[Dict, ConversationBufferMemory]:
        """
        Sets up memory for the open ai functions agent.
        :return a tuple with the agent keyword pairs and the conversation memory.
        """
        conversational_memory = ConversationBufferWindowMemory(
            memory_key='chat_history',
            k=max_history,
            return_messages=True
        )
        return conversational_memory

    def _handle_error(self, error) -> str:
        return str(error)[:50]


    #https://www.google.com/search?client=safari&rls=en&q=langchian+call+custom+tool&ie=UTF-8&oe=UTF-8#fpstate=ive&vld=cid:56a7070f,vid:q-HNphrWsDE,st:0
    #Need to figure out how to get LLM to pull parameters to send to the tool!
    def setup_agent(self, max_iterations=3) -> AgentExecutor:
        """
        Sets up the tools for a function based chain.
        We have here the following tools:
        - image2text
        """
        cfg = self.config()
        tools = [
            Tool(name=Image_to_Text().name,
            func=Image_to_Text()._run,
            description=Image_to_Text().description
            ),

        Tool(name="LLM",
                func=cfg,
                description="Language model to answer qurries or format text"
                )
        ]

        conversational_memory = self.setup_memory()

        sys_msg = """Assistant is a sophisticated language model developed by OpenAI.

Designed to handle a diverse array of tasks, Assistant excels in answering questions, providing detailed explanations, and engaging in meaningful discussions across various subjects. As a language model, Assistant generates human-like text based on the input it receives, ensuring coherent and contextually relevant responses.

However, it's important to note that Assistant does not have the inherent ability to convert images to text. When faced with image-related tasks, such as extracting text from images, Assistant seamlessly integrates with specialized tools. It refers to these external tools to perform image-to-text conversion tasks rather than attempting to do so directly.

Continuously learning and evolving, Assistant leverages its extensive training on text data to offer accurate and informative responses. Whether you have specific inquiries or seek insights on diverse topics, Assistant is here to assist, leveraging external tools when needed for image-related tasks.
            """

        return initialize_agent(
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,#agent='chat-conversational-react-description',#Chat convo causes error....
            tools=tools,
            llm=cfg,
            verbose=True,
            max_iterations=3,
            system_message=sys_msg,
            early_stopping_method='generate',
            memory=conversational_memory,
            handle_parsing_errors=self._handle_error
        )

if __name__ == '__main__':
    
    agent = ConfigAgent().setup_agent(max_iterations=3)
    print(dir(agent.run))
    pdf_path = "/Users/viktorciroski/Documents/Github/Forestry_Student/Test_Results/FOR205_Final Exam_Fall 2014_ Sample A 2.pdf"

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Get the text from this file {pdf_path}"}
    ]
    query=f"Get the text from this file {pdf_path}"
    output = agent.run(messages)

    print("OUTPTU")
    print(output)
    #agent.run(HumanMessage(content=query))