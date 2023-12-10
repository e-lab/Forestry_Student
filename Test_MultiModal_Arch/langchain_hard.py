from langchain.agents import AgentType, Tool, initialize_agent
from langchain.llms import OpenAI
from langchain.tools import BaseTool
from transformers import pipeline, AutoFeatureExtractor

from transformers import pipeline, GenerationConfig
from huggingface_hub import HfApi

import openai
from PIL import Image
import requests
from io import BytesIO
import pandas as pd
import os
import fitz
import string
import json
import csv
import sqlite3
import time
from tqdm import tqdm

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
            
    def get_text(self,image):
        response = pipe(
            image, 
            max_new_tokens=pipe.tokenizer.model_max_length
        )
        text = response[0].get('generated_text', '')
        return text 
    
    def check_urls(self, urls):
        if type(urls) == str:
            urls = [urls]
        elif type(urls) == list:
            pass
        else:
            return "Please provide us with a list of the URLS"
        return urls


    def _run(self, urls):
        results = []
        urls = self.check_urls(urls)
        
        for url in urls:
            images = self._process_url_or_path(url)
            if images:
                for i, image in enumerate(images):
                    text = self.get_text(image)
                    results.append(text)
                    #print(len(results))
                break
        
        return results if results else None
    

class QuestionTypeGenerator(BaseTool):
    def __init__(self):
        super().__init__(
            name="question_type_generator",
            func=self._run,
            description=("""This tool processes raw text from exams or homework and generates a question type dictionary.
            Provide the URLs as a list, and specify the desired format for the Python dictionary.
            The output includes a list of questions and their corresponding types"""),
        )

    

    def remove_punctuation(self, input_string):
        # Create a translation table for removing punctuation excluding underscore
        exclude_chars = string.punctuation.replace("_", "")
        translator = str.maketrans("", "", exclude_chars)
        
        # Use the translation table to remove punctuation
        result_string = input_string.translate(translator)
        
        return result_string


    def _run(self, urls, dict_format):
        file_name = "question_type.csv"
        image_to_text_tool = ImageToTextTool()
        raw_text_list = image_to_text_tool._run(urls)

        # Initialize the language model agent
        #mrkl = Init_Agent().get_simple_llm()
        #mrkl = Init_Agent().get_agent(agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        #                              tools=[])

        # Construct the message with raw text and dictionary format
        # Example of how the JSON output might look:
        example_json_output = '''
        {
        "questions": [
            {"question": "What is the capital of France?", "question_type": "general"},
            {"question": "Who wrote Hamlet?", "question_type": "trivia"}
        ],
        "question_types": [
                            "Objective Questions",
                            "Short Answer Questions",
                            "Matching Questions",
                            "Essay and Long Answer Questions",
                            "Problem-Solving Questions",
                            "Case Study or Scenario-Based Questions",
                            "Definition and Identification Questions",
                            "Diagram-Based Questions",
                            "Reading and Comprehension",
                            "Interpretation Questions",
                            "Comparison and Evaluation Questions",
                            "Application and Creative Writing",
                            "Research-Based Questions",
                            "Vocabulary and Synthesis Questions",
                            "Critical Thinking Questions",
                            "Ethical Dilemma and Historical Analysis",
                        ]

        }
        '''

        data = []
        for raw_text in raw_text_list:
            formated_text = self.remove_punctuation(raw_text)
            #print(formated_text)
            #print("_____")
            #print(f"Desired Dict Format {dict_format}")
            msg = f"{example_json_output} {formated_text} do not provide the answers only a string in the dictionary format provided! and remove any portions that do not seem like quesitons"
            sys_msg = f"The formatted text should follow the format of the example JSON output: {example_json_output}. Please provide the answers only as a string in the dictionary format. Additionally, remove any portions that do not seem like questions. YOU SHOULD NEVER ATTEMPT TO ANSWER THE QUESTION ONLY"

            # Run the language model to generate the question type dictionary
            dict_out = Init_Agent().comp(sys_msg, msg)
            data_dict = json.loads(dict_out)
            """
            print("OUTPTU")
            print(dict_out)
            print("____")
            print("DICT VARIBLE")
            
            # Now, data_dict is a Python dictionary that you can work with
            print(data_dict)
            print("_____")
            # Update the master dictionary with key-value pairs from data_dict
            """
            data.append(data_dict['questions'])


        data_dict = {"question": [], "type": []}

        for part in data:
            for qt in part:
                try:
                    # Check if 'question' and 'question_type' keys exist
                    if 'question' in qt and 'question_type' in qt:
                        # Check if 'question' value is a string
                        if isinstance(qt['question'], str):
                            data_dict['question'].append(qt['question'])
                            data_dict['type'].append(qt['question_type'])
                        else:
                            print(f"Warning: Question in {qt} is not a string.")
                    else:
                        print(f"Warning: Missing 'question' or 'question_type' keys in {qt}")
                        
                    # Check if 'answers' key exists and its value is a list
                    if 'answers' in qt and isinstance(qt['answers'], list):
                        # Extend the lists in data_dict with each answer
                        data_dict['question'].extend(qt['answers'])
                        data_dict['type'].extend([qt.get('question_type', '')] * len(qt['answers']))
                        
                except Exception as e:
                    print(f"Error processing {qt}: {e}")

        try:
            with open(file_name, 'w', newline='', encoding='utf-8') as csvfile:
                csvwriter = csv.writer(csvfile)

                # Write header
                csvwriter.writerow(['question', 'type'])

                # Write data
                for i in range(len(data_dict['question'])):
                    csvwriter.writerow([data_dict['question'][i], data_dict['type'][i]])

            print(f"Data successfully saved to {file_name}")

        except Exception as e:
            print(f"Error saving data to {file_name}: {e}")

        return f"Qutestion type dictionary was saved to {file_name}"
    
class SimilarityQA(BaseTool):
    def __init__(self):
        super().__init__(
            name="similarity_question_answering",
            func=self._run,
            description=("""The Similarity Question Answering tool is designed to process a diverse range of questions, 
                         including those from exams, homework, or general student inquiries. It leverages a comprehensive 
                         vector store database containing previously uploaded educational materials. The tool aims to generate 
                         context-aware responses by analyzing the provided questions in relation to the existing corpus of educational content.

                         To use the tool, you can either provide a list of questions directly or specify a path to a CSV file containing 
                         the questions. If a data path is provided, the tool will read the CSV file and convert it into a list of queries. 
                         Additionally, you have the option to save the generated documents, enhancing the learning material in the vector store database.

                         This tool utilizes an advanced language model to ensure precise and coherent responses. Whether you are a student seeking 
                         clarification or an educator looking to augment teaching resources, the Similarity Question Answering tool streamlines the 
                         process of extracting relevant information from the stored educational content.
                      """),
        )

    def load_database(self, database_name = 'trees.db'):
        self.conn = sqlite3.connect(database_name)
        self.cursor = self.conn.cursor()

    def _run(self, database_name='courses.db', data_path=None, queries=None, save_doc_flag=False, output_file='EXAM_ANSWERS.txt', top_k=1):
        if queries is None and data_path is None:
            return "Please provide us with a question or a path to questions."
        if queries is None:
            queries = []
            df = pd.read_csv(data_path)
            print(df.head())
            # Format CSV to a list of queries
            for index, row in df.iterrows():
                # Access row values using column names
                msg = f"{row['question']} answer in a {row['type']} format"
                queries.append(msg)
            save_doc_flag = True

        # self.load_database(database_name=database_name)

        llm = Init_Agent().get_simple_llm_pipeline()
        answers = []  # List to store answers
        for query in tqdm(queries, desc="Processing queries", unit="query"):
            """Add logic for looking up the most similar context to answer the question from the database"""
            sys_msg = ""
            response = llm(f"{sys_msg} {query}", top_k=top_k)
            answers.append(response)
            #time.sleep(0.5) 


        if save_doc_flag:
            # Write answers to the output file
            with open(output_file, 'w') as file:
                for ans in answers:
                    file.write(ans + '\n')
        
        # Optionally, you can return the list of answers or perform other actions as needed.
        return answers


class Init_Agent():
    def __init__(self):
        super().__init__()

    def get_simple_llm_pipeline(self, model_id = 'meta-llama/Llama-2-7b-chat-hf'):
        
        

        # Set up the generation config
        gen_config = GenerationConfig.from_pretrained(
            model_id,
            token=hf_token
        )
        gen_config.max_new_tokens = 2000
        gen_config.temperature = 1e-10

        # Set up the Hugging Face pipeline for text generation
        pipe = pipeline(
            task="text-generation",
            model=model_id,
            return_full_text=True,
            generation_config=gen_config,
            device_map='auto',
            repetition_penalty=1.1
        )
        return pipe 
    
    def comp(self, sys_msg, PROMPT, MaxToken=3000, outputs=3): 
        # using OpenAI's ChatCompletion module that helps execute  
        # any tasks involving text  
        PROMPT = f"You: {PROMPT}\nAI:"
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are a helpful assistant. {sys_msg}"},
                {"role": "user", "content": PROMPT},
            ],
            max_tokens=MaxToken
        )
        
        # Extract the assistant's reply from the response
        assistant_reply = response['choices'][0]['message']['content'].strip()
        
        return assistant_reply



    def get_agent(self, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, tools = [QuestionTypeGenerator()]):
        

        # Initialize the agent
        llm = OpenAI(temperature=0)
        self.tools = tools
        if len(self.tools) <1:
            self.agent = initialize_agent(llm=llm, 
                                          agent=agent_type, 
                                          verbose=True)   
        else: 
            self.agent = initialize_agent(self.tools,
                                        llm,
                                        agent=agent_type, 
                                        verbose=True)
        return self.agent

hf_token = "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"
os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.environ["OPENAI_API_KEY"]
api = HfApi()
api.token = hf_token
if __name__ == "__main__":

    
    # Initialize the tool
    image_to_text_tool = ImageToTextTool()
    gen_QT = QuestionTypeGenerator()
    sim_QA = SimilarityQA()

    sim_QA._run(data_path="question_type.csv")
    """

    tools = [gen_QT, sim_QA]
    mrkl = Init_Agent().get_agent(agent_type=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                                  tools = tools
                                  )
    
    # Provide a PDF file path
    pdf_path = "/Users/viktorciroski/Documents/Github/Forestry_Student/Test_Results/FOR205_Final Exam_Fall 2014_ Sample A 2.pdf"

    # Run the agent to get text from the PDF file
    message = f"Extract text from the specified PDF file located at {pdf_path}. After completing the text extraction, generate a JSON object with two lists: one for questions and the other for their corresponding question types. The output format should be a JSON object containing 'questions' and 'question_types' arrays."

    

    output = mrkl.run(message)
    print("________")
    print(output)
    """
