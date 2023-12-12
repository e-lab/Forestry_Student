from torch import cuda, bfloat16
import transformers
"""
model_id = 'meta-llama/Llama-2-7b-chat-hf'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# begin initializing HF items, you need an access token
hf_auth = "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"
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    device_map='auto',
    use_auth_token=hf_auth
)

# enable evaluation mode to allow model inference
model.eval()

print(f"Model loaded on {device}")

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

stop_list = ['\nHuman:', '\n```\n']

stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
stop_token_ids

import torch

stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
stop_token_ids

from transformers import StoppingCriteria, StoppingCriteriaList

# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])

generate_text = transformers.pipeline(
    model=model, 
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    stopping_criteria=stopping_criteria,  # without this model rambles during chat
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # max number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)
res = generate_text("Explain me the difference between Data Lakehouse and Data Warehouse.")
print(res[0]["generated_text"])

from langchain.llms import HuggingFacePipeline

llm = HuggingFacePipeline(pipeline=generate_text)

# checking again that everything is working fine
res = llm(prompt="Explain me the difference between Data Lakehouse and Data Warehouse.")
print(res)
print(print(res[0]["generated_text"]))

from langchain.text_splitter import RecursiveCharacterTextSplitter
"""
def split_to_chunks(data, chunk_size=1000, chunk_overlap=100):
    """
    Splits a given data string into chunks of specified size with an optional overlap.

    Parameters:
    - data (str): The input string to be split into chunks.
    - chunk_size (int, optional): The desired size of each chunk. Defaults to 1000.
    - chunk_overlap (int, optional): The number of overlapping characters between adjacent chunks. Defaults to 100.

    Returns:
    - list of str: A list containing the resulting chunks.
    """

    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size - chunk_overlap)]
    return chunks



        
import fitz  # PyMuPDF
def read_pdf(file_path):
    """Read text content from a PDF file using PyMuPDF."""
    pdf_text = ''
    try:
        pdf_document = fitz.open(file_path)
        num_pages = pdf_document.page_count
        for page_num in range(num_pages):
            page = pdf_document[page_num]
            pdf_text += page.get_text()
    except Exception as e:
        print(f"Error reading PDF: {e}")
    finally:
        if 'pdf_document' in locals():
            pdf_document.close()
    return pdf_text
import glob
import os
print("Getting PDFS...")
# Directory where the PDF files are located
pdf_directory = "pdfs/"
# Use glob to get a list of PDF files in the directory
pdf_files = glob.glob(os.path.join(pdf_directory, '*.pdf'))
file_path_list = []
# Add the file paths to the file_path_list
file_path_list.extend(pdf_files)
chunks = ''
print ("Embbeding Data...")

for file_path in file_path_list:
    file_contents = read_pdf(file_path)
    chunks += file_contents 
    #add_embeddings(file_contents)#, file.filename, class_name, topic_name, database_name)

print(chunks)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
global vectorstore
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
all_splits=split_to_chunks(chunks, chunk_size=1000, chunk_overlap=100)
# storing embeddings in the vector store

vectorstore = FAISS.from_texts(all_splits, embeddings)

from langchain.chains import ConversationalRetrievalChain

chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)
