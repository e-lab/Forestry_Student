import streamlit as st
import os, re, json 
import base64
import extra_streamlit_components as stx
from annotated_text import annotated_text
import datetime 
from langchain_core.messages import AIMessage, HumanMessage

@st.cache_resource(experimental_allow_widgets=True) 
def get_manager():
    return stx.CookieManager()

class CookieManager: 
  def __init__(self, cookie_name = 'messages'):
    self.manager = get_manager()
    self.cookie_name = cookie_name

  def __call__(self): 
    _ = self.manager.get_all()
  
  def get(self):
    return self.manager.get(cookie=self.cookie_name)
  
  def set(self, value):
    self.manager.set(self.cookie_name, value)
  
  def delete(self):
    self.manager.delete(cookie=self.cookie_name)

class Chat_UI: 
  def __init__(self, pipeline): 
    self.pipeline = pipeline
    self.cookie_manager = CookieManager() 

  def render(self):
    self.chat() 

  def initiate_memory(self): 
    history = self.get_messages()

    if not history:
      st.session_state['messages'] = [{"role": "assistant", "content": "Hello! The name's euGenio. I'm here to help you with your pipeline. Ask me a question!"}]
    else: 
      st.session_state['messages'] = history
  
  def append(self, message:dict): 
    st.session_state['messages'].append(message)

  def __call__(self): 
    self.cookie_manager()
    # Instantiates the chat history
    self.initiate_memory() 
    self.load_memory()

    # Load's the text tab
    self.load_chatbox()
        
  def load_chatbox(self):
    user_input = st.text_input("*Got a question?*", help='Try to specify keywords and intent in your question!', key="text", on_change=self.handle_query)

    if st.button('Delete History', use_container_width=True, type='primary'):
      self.delete_messages() 

  def load_memory(self): 
    messages = st.session_state['messages'] 
    if messages: 
      for message in messages :
        role = message["role"]
        content = message["content"]
        
        with st.chat_message(role):
          if type(content) == dict and role == 'assistant': 
            with st.expander("Thought Process!", expanded=True): 
              st.json(content)
            
          else: 
            st.markdown(content)

  def format_history(self): 
    messages = st.session_state['messages']

    if messages:
      formatted = [] 
      for message in messages[1:]:
        if message['role'] == 'user': 
          formatted.append(HumanMessage(content=str(message['content'])))
        else: 
          formatted.append(AIMessage(content=str(message['content'])))
      return formatted
    else: 
      return []

  def handle_query(self):
    text = st.session_state["text"] 
    st.session_state["text"] = ""

    user_message = {"role": "user", "content": text}  
    self.append(user_message)

    with st.chat_message("user"):
        st.markdown(text)

    with st.chat_message("assistant"):
        idx, tool = 0, None

        with st.spinner('Thinking...'): 
          results = self.pipeline.run(query=text, chat_history=self.format_history())
          
        st.markdown(results['output'])
        
        with st.expander("Sources", expanded=False): 
          result = self.pipeline.get_sources(f"{text}: {results['output']}")
          print(result)

        idx += 1

    assistant_message = {"role": "assistant", "content": {key: value for key, value in results.items() if key != 'chat_history'}}

    self.append(assistant_message)
    self.store_messages(user_message, assistant_message)

  def store_messages(self, user_message, assistant_message): 
    past = self.cookie_manager.get()

    if past: 
      if user_message not in past and assistant_message not in past:
        past.append(user_message)
        past.append(assistant_message)
        self.cookie_manager.set(past)
    else:
      self.cookie_manager.set(st.session_state.messages)
  
  def get_messages(self): 
    return self.cookie_manager.get()
  
  def delete_messages(self): 
    self.cookie_manager.delete()
    self.initiate_memory()

  def _generate_images(self, document): 

    images = [self.highlight_bbox_in_pdf(document['id'], document['page_num'], (document['xmin'], document['ymin'], document['xmax'], document['ymax']))]
    
    images_markdown = self.create_markdown_with_images(images)

    return images_markdown

  def highlight_bbox_in_pdf(self, pdf_path, page_number, bbox):
      doc = fitz.open(pdf_path)
      page = doc.load_page(page_number)
      
      pix = page.get_pixmap()
      img = Image.open(io.BytesIO(pix.tobytes()))
      
      draw = ImageDraw.Draw(img)
      draw.rectangle(bbox, outline="red", width=2)
      
      img_buffer = io.BytesIO()
      img.save(img_buffer, format="PNG")  # Save the modified image to the buffer
      encoded_image = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
      
      doc.close()
      
      return encoded_image
  
  def create_markdown_with_images(self, images):
    images_html = ""
    for base64_image in images:
        img_html = f'<img src="data:image/png;base64,{base64_image}" style="display: block; margin-left: auto; margin-right: auto; width: 80%;">'
        images_html += img_html + "<br>"
    return images_html

class CookieTester: 
  def __init__(self): 
    self.cookie = None