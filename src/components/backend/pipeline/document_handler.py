import numpy as np 
import fitz
import requests
import time as time
import uuid 
import streamlit as st 
import base64

class Document_Handler: 
  def __init__(self):
    pass 

  def __call__(self, bytes_array):
    return self.extract_and_chunk(bytes_array) 

  def semantic_chunking(self, text, chunk_size=200, overlap=50):
    chunks = []
    current_chunk = ""

    if type(text) == str: 
      words = text.split()
    elif type(text) == list: 
      words = text

    for word in words:
        current_chunk += (word + " ")
        if len(current_chunk) >= chunk_size:
            period_pos = current_chunk.rfind('. ')
            if period_pos != -1 and period_pos + 1 < len(current_chunk):
                chunks.append(current_chunk[:period_pos + 1])
                current_chunk = current_chunk[max(period_pos + 1 - overlap, 0):]
            else:
                chunks.append(current_chunk.strip())
                current_chunk = ""

    if len(current_chunk) > chunk_size // 2:
        chunks.append(current_chunk.strip())

    return chunks

    
  def extract_and_chunk(self, bytes_array):
    doc = fitz.open(stream=bytes_array, filetype="pdf")

    text_blocks = []
    id = str(uuid.uuid4())
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]

        for b in blocks:
            if "lines" in b: 
                bbox = fitz.Rect(b["bbox"])
                text = " ".join([" ".join([span["text"] for span in line["spans"]]) for line in b["lines"]])
                
                if len(text.split()) > 100:
                    chunks = self.semantic_chunking(text)
                else:
                    chunks = [text]

                for chunk in chunks:
                    text_blocks.append((id, page_num, bbox.x0, bbox.y0, bbox.x1, bbox.y1, chunk))
        
        print('here')

    doc.close()
    return text_blocks
