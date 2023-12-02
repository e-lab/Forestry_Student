from langchain.agents import AgentType, Tool, initialize_agent
from langchain.llms import OpenAI
from langchain.tools import BaseTool
from transformers import pipeline, AutoFeatureExtractor
from PIL import Image
import requests
from io import BytesIO
import os
import fitz

class ImageToTextTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="image_to_text",
            func=self._run,
            description=("""This tool utilizes the Nougat model by Meta to extract text from images, PDFs, 
            or URLs by converting them to image files. Ideal for digitizing text from scanned 
            documents, photos, or online sources, providing high-quality text output.
            To use this tool, you must provide the URL parameter as a list of URLs"""),
        )
        global pipe #Trying to get around self.pipe error....
        pipe = pipeline(
            task='image-to-text', 
            model='facebook/nougat-base', 
            feature_extractor=AutoFeatureExtractor,
        )
        
    def _process_url_or_path(self, url):
        if url.startswith(('http://', 'https://')):
            response = requests.get(url)
            response.raise_for_status()
            if response.headers['content-type'].startswith('image'):
                return [Image.open(BytesIO(response.content))]
            else:
                pdf_document = fitz.open("pdf", response.content)
                return [
                    Image.frombytes("RGB", [page.get_pixmap().width, page.get_pixmap().height], page.get_pixmap().samples)
                    for page in pdf_document
                ]
        else:
            if os.path.isfile(url):
                if url.lower().endswith('.pdf'):
                    pdf_document = fitz.open(url)
                    return [
                        Image.frombytes("RGB", [page.get_pixmap().width, page.get_pixmap().height], page.get_pixmap().samples)
                        for page in pdf_document
                    ]
                else:
                    return [Image.open(url)]
            else:
                return []

    def _run(self, urls):
        results = []
        if type(urls) == str:
            urls = [urls]
        elif type(urls) == list:
            pass
        else:
            return "Please provide us with a list of the URLS"
        
        for url in urls:
            images = self._process_url_or_path(url)
            if images:
                for i, image in enumerate(images):
                    response = pipe(
                        image, 
                        max_new_tokens=pipe.tokenizer.model_max_length
                    )
                    text = response[0].get('generated_text', '')
                    results.append(text)
                    print(len(results))
        print(results)
        return results if results else None

if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = "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"

    # Initialize the tool
    image_to_text_tool = ImageToTextTool()

    # Initialize the agent
    llm = OpenAI(temperature=0)
    tools = [image_to_text_tool]
    mrkl = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    # Provide a PDF file path
    pdf_path = "/Users/viktorciroski/Documents/Github/Forestry_Student/Test_Results/FOR205_Final Exam_Fall 2014_ Sample A 2.pdf"

    # Run the agent to get text from the PDF file
    messages = f"Get the text from this file {pdf_path}"
    output = mrkl.run(messages)
    print("________")
    print(output)
