import os 
import os
from openai import OpenAI

import json 
import pandas as pd
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import numpy as np
import torch

from transformers import NougatProcessor, VisionEncoderDecoderModel

# Use a model from OpenAI (assuming "text-embedding-ada-002" exists for this example)
model_name="gpt-3.5-turbo"
os.environ["OPENAI_API_KEY"] = "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"
device = "cuda" if torch.cuda.is_available() else "cpu"

from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

client =  OpenAI(api_key=os.environ["OPENAI_API_KEY"])






def get_nougart():
    processor = NougatProcessor.from_pretrained("facebook/nougat-base")
    model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")
    model.to(device)
    return model, processor

def read_pdf_with_images(file_path, model, processor):
    """Read text content from a PDF file and display images using PyMuPDF."""
    pdf_text = ''
    pdf_document = None

    text_list = []

    try:
        pdf_document = fitz.open(file_path)
        num_pages = pdf_document.page_count

        for page_num in range(num_pages):
            print(f"Completion Percentage:\t{(page_num/num_pages)*100}%") #Update with real progress bar
            # Display image
            page = pdf_document[page_num]
            pixmap = page.get_pixmap()
            width, height = pixmap.width, pixmap.height
            img_array = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape((height, width, -1))
            pixel_values = processor(img_array, return_tensors="pt").pixel_values
            #print("TEST")


            # Display the image
            #plt.imshow(img_array, cmap="gray")
            #plt.axis("off")
            #plt.title(f"Page {page_num + 1}")
            #plt.show()

            # Read text content
            outputs = model.generate(
                pixel_values,
                min_length=1,
                max_new_tokens=3584,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
            )
            sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            sequence = processor.post_process_generation(sequence, fix_markdown=False)
            #print(sequence)
            text_list.append(sequence)
            #pdf_text += page.get_text() #Change to nougart to extraxt text 

    except Exception as e:
        print(f"Error reading PDF: {e}")
    finally:
        if pdf_document is not None:
            pdf_document.close()

    return text_list



def chat_with_openai(prompt):
    """
    Sends the prompt to OpenAI API using the chat interface and gets the model's response.
    """
    message = {
        'role': 'user',
        'content': prompt
    }

    response = client.chat.completions.create(model=model_name,
    messages=[message])

    # Extract the chatbot's message from the response.
    # Assuming there's at least one response and taking the last one as the chatbot's reply.
    chatbot_response = response.choices[0].message.content
    return chatbot_response.strip()

def main(file_path):
    """
    update this to read page by page form a textbook and generate qa pairs to a dataset. we'll need to test to ensure the output format is right 
    """
    forestry_dict = {"questions":[], "answers":[]}
    missed_text = []
    file_name = file_path.split("/")[-1].split(".")[0]
    #model, processor = get_nougart()
    #text_list = read_pdf_with_images(file_path, model, processor)
    #pd.DataFrame({"OCR":text_list}).to_csv(f"{file_name}_OCR.csv")
    counter = 0
    text_list = pd.read_csv(f"{file_name}_OCR.csv")
    text_list = list(text_list['OCR'])
    while len(text_list)>1:
        
        for paragraph in text_list:
            question = """Generate 10 forestry based questions answer pairs from the following text. Format the output json for question and answer columns each row should be a new qa pair for example output should be {[
                            {
                            "question": "What are some characteristics of Virginia pine?",
                            "answer": "Virginia pine is intolerant of shade, has a moderate growth rate, and is a hard pine."
                            },
                            {
                            "question": "How tall does Virginia pine grow?",
                            "answer": "Virginia pine grows to a height of 40 to 60 feet."
                            },
                            {
                            "question": "What is the diameter of a mature Virginia pine?",
                            "answer": "A mature Virginia pine attains a diameter of 1 to 1.5 feet."
                            },
                            {
                            "question": "What is Virginia pine often grown for in the southern states?",
                            "answer": "Virginia pine is often grown for Christmas trees."
                            }
                            }
                        ]}"""
            #paragraph = "Virginia pine is intolerant of shade, has a moderate growth rate, and is a hard pine. It grows to a height of 40 to 60 feet, and attains a diameter of 1 to 1.5 feet. Virginia pine is often grown for Christmas trees in the southern states. A good identification feature of this species is the abundance of cones present, even persisting on dead branches. The crown may become flat topped if grown in the open. Virginia pine is susceptible to wind throw in exposed locations."
            user_input = f"{question} {paragraph}"
            try:
                response = chat_with_openai(user_input)  # Pass user_input as an argument
                print(f"Chatbot: {response}")
            

                json_dict = json.loads(response)
                for i in range(len(json_dict)):
                    q = json_dict[i]['question']
                    a = json_dict[i]['answer']
                    forestry_dict['questions'].append(q)
                    forestry_dict['answers'].append(a)
                pd.DataFrame(forestry_dict).to_csv(f"{file_name}_forestry_QA_{counter}.csv")
            except:
                missed_text.append(paragraph)
                pd.DataFrame({"missed":missed_text}).to_csv(f"{file_name}_forestry_QA_MISSED_REDO.csv")

            
        print(pd.DataFrame(forestry_dict))
        counter += 1
        text_list = pd.read_csv(f"/Users/viktorciroski/Documents/Github/Forestry_Student/datamine_textbox_qa/NHLA Rules for The Measurement and Inspection of Hardwood and Cypress_forestry_QA_MISSED_REDO.csv")
        print(f"Len of missed data {len(text_list)}")
        text_list = list(text_list['missed'])
        missed_text = []
def get_embedding(text, model="text-embedding-ada-002"):
   text = str(text).replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding #Can we embedd data with LLama 

def append_similarity(file_path):
    df = pd.read_csv(file_path)
    #df = df.iloc[0:10]
    print(df.head())
    df['ada_embedding'] = df['questions'].apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))

    ada_embeddings = np.array(df['ada_embedding'].to_list())

    similary_score = []
    for i in range(len(df)):
        q1 = ada_embeddings[i].reshape(1, -1)
        q2 = np.delete(ada_embeddings, i, axis=0)  # Exclude q1 from q2 values
        similarities = cosine_similarity(q1, q2)[0]

        max_similarity_index = np.argmax(similarities)
        max_similarity = similarities[max_similarity_index]
        similary_score.append(max_similarity)

        print(f"For question {i}, max similarity score is {max_similarity} with question {max_similarity_index}.")

    df['similarity_score'] = similary_score
    df = df.drop('ada_embedding', axis=1)


    df.to_csv('forestry_embedded_reviews.csv', index=False)

def filter_forestry_questions(file_path, similarity_TOL=1):
    df = pd.read_csv(file_path)
    print(f"Original Length {len(df)}")
    df = df[df['similarity_score']<similarity_TOL]
    print(f"Similarity Filiter Length {len(df)}")

    responses = []
    for i in range(len(df)):
        question = "Does this question relate to forestry specific topics. If yes respond with a 1 if no respond with a 0 and only respond with a 1 or 0"
        paragraph = df['questions'].iloc[i]
        user_input = f"{question} {paragraph}"
        #try:
        response = chat_with_openai(user_input)  # Pass user_input as an argument
        print(f"Chatbot: {response}")
        responses.append(response)
        #except:
        #    responses.append("REDO")

    print(responses)
    df["forestry_question"] = responses
    df = df[df["forestry_question"]==1]
    df.to_csv('forestry_type_questions.csv', index=False)
    print(f"Forestry Type Questions Filter Length {len(df)}")








    


if __name__ == "__main__":
    file_path="/Users/viktorciroski/Desktop/pdfs/NHLA Rules for The Measurement and Inspection of Hardwood and Cypress.pdf"
    main(file_path) 
    model, processor = get_nougart()
    read_pdf_with_images(file_path="/Users/viktorciroski/Desktop/pdfs/NHLA Rules for The Measurement and Inspection of Hardwood and Cypress.pdf", model=model, processor=processor)
    append_similarity("/Users/viktorciroski/Documents/Github/Forestry_Student/datamine_textbox_qa/NHLA Rules for The Measurement and Inspection of Hardwood and Cypress_forestry_QA_144.csv")
    filter_forestry_questions(file_path="forestry_embedded_reviews.csv", similarity_TOL=0.999999)
    df = pd.read_csv("forestry_type_questions.csv")
    df = df[df["forestry_question"]==1]
    df = df[df['similarity_score']<0.99999999]
    df.to_csv('forestry_type_questions.csv', index=False)
    print(df['forestry_question'].unique())
    print(f"Forestry Type Questions Filter Length {len(df)}")