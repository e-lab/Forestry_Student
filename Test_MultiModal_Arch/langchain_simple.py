from langchain.agents import AgentType, Tool, initialize_agent
from langchain.llms import OpenAI
import os 

os.environ["OPENAI_API_KEY"] = "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" # Get it at https://beta.openai.com/account/api-keys

def multiplier(a=0, b=0):
    print(f"a\t{a}")
    print("b\t{b}")
    return a * b


def parsing_multiplier(string):
    a, b = string.split(",")
    return multiplier(int(a), int(b))

def image_to_text(url):
    print(url)
    return url

llm = OpenAI(temperature=0)
tools = [
    Tool(
        name="image_to_text",
        func=image_to_text,
        description=(f"""This tool utilizes the Nougat model by Meta to extract text from images, PDFs, 
    or URLs by converting them to image files. Ideal for digitizing text from scanned 
    documents, photos, or online sources, providing high-quality text output.
    To use this tool you must provide the URL prarameter as a list of urls""")
              )
]
mrkl = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

pdf_path = "/Users/viktorciroski/Documents/Github/Forestry_Student/Test_Results/FOR205_Final Exam_Fall 2014_ Sample A 2.pdf"

messages = f"Get the text from this file {pdf_path}"
output = mrkl.run(messages)
#mrkl.run("What is 3 times 4")
print(output)