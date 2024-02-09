import numpy as np 
import fitz
import requests
import time as time
import uuid 
import streamlit as st
import base64, glob, os

class Document_Handler: 
  def __init__(self):
    pass 

  def __call__(self, bytes_array):
    return self.extract_and_chunk(bytes_array) 

  def smart_chunking(self, text, chunk_size=200, overlap=50):
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

    text_blocks = ""
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]

        for b in blocks:
            if "lines" in b: 
                text = " ".join([" ".join([span["text"] for span in line["spans"]]) for line in b["lines"]])
                text_blocks += text 

    doc.close()
    return text_blocks

  def load(self, paths):
    total = [] 
    for path in paths: 
      if path.endswith(".pdf"):
        with open(path, "rb") as f: 
          total = [self.extract_and_chunk(f.read())]
      elif path.endswith(".txt"): 
        with open(path, "r") as f: 
          total.extend(self.smart_chunking(f.read()))
    return total 
    
  def __len__(self):
    return len(glob.glob(f"{os.environ['TMP']}/*.pdf") + glob.glob(f"{os.environ['TMP']}/*.txt") + glob.glob(f"{os.environ['TMP']}/*.csv"))