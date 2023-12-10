from langchain.tools import BaseTool
from transformers import pipeline, AutoFeatureExtractor

from PIL import Image
import requests
from io import BytesIO
import os
import fitz

pipe = pipeline(
                task='image-to-text', 
                model='facebook/nougat-base', 
                feature_extractor=AutoFeatureExtractor,
            )

class Image_to_Text(BaseTool):
    name = "image_to_text"
    description = (f"""This tool utilizes the Nougat model by Meta to extract text from images, PDFs, 
    or URLs by converting them to image files. Ideal for digitizing text from scanned 
    documents, photos, or online sources, providing high-quality text output.
    To use this tool you must provide the URL prarameter as a list of urls""")

    
   

    def _process_url_or_path(self, url:list[str] ):
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
                #print("Invalid file path or URL.")
                return []

    def _run(self, urls):
        print(urls)
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
                    #print(f"Text for {url} - Page {i + 1}:\n{text}")
                    results.append(text)
                    print(len(results))
        print(results)
        return results[0]

    async def _arun(self, urls):
        raise NotImplementedError("custom_search does not support async")

# Example usage:
if __name__ == '__main__':
    urls = [
        "/Users/viktorciroski/Documents/Github/Forestry_Student/Test_Results/FOR205_Final Exam_Fall 2014_ Sample A 2.pdf"
    ]
    
    tool = Image2Text()
    results = tool._run(urls)
    print("All Results:", results)
