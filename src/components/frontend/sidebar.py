import streamlit as st
import base64, os 
import time 
import io

class Sidebar: 
  def __init__(self, pipeline): 
    self.pipeline = pipeline

  def __call__(self):
    with st.sidebar: 
      st.markdown(
          """
          <style>
              [data-testid=stSidebar] [data-testid=stImage]{
                  text-align: center;
                  display: block;
                  margin-left: auto;
                  margin-right: auto;
                  border-radius: 50%;
                  margin-top: -75px;
              }
          </style>
          """, unsafe_allow_html=True
      )
      
      
      st.image('assets/eugenie.png', width=250)

    disabled = True 
    if 'api_key' not in st.session_state:
      disabled = False
    key = st.sidebar.text_input('', placeholder ='Input your OpenAI API Key: ', type='password', label_visibility='hidden', key='api_key_input', disabled=disabled)
    if key: 
      st.session_state['api_key'] = key
      st.sidebar.success('API Key Successfully Added!')
    st.sidebar.divider() 

    self._upload_widget() 
    self._show_tools()

  def _upload_widget(self): 

    upload_expander = st.sidebar.expander("File Uploader", expanded=True)
    with upload_expander: 
      pdf_docs = st.file_uploader('Select Files to Upload', accept_multiple_files=True, type=['pdf', 'txt', 'png', 'jpg'])
      if st.button('Start Upload'): 
        for pdf in pdf_docs:
          file_details = {'Filename': pdf.name, 'FileType': pdf.type, 'FileSize': pdf.size}

          progress_text = 'Checking File...'
          my_bar = st.progress(0, text=progress_text)
          percent_complete = 0

          if pdf.type == "application/pdf":
            percent_complete += 50 
            my_bar.progress(percent_complete, text=progress_text)
            progress_text = 'Processing File...'

            status = self.pipeline.add(pdf.read())

            print(status)

            percent_complete += 50
            my_bar.progress(percent_complete, text="Finalizing...")
            st.success(f'File Successfully Processed!')
            my_bar.empty()
          del my_bar
          
    st.session_state['documents'] = True

  def _show_tools(self): 
    tools = st.sidebar.expander("Tools", expanded=True)
    with tools: 
      options = st.selectbox(
        'List of available tools: ',
        ('Chroma-DB', 'Web-Search', 'arXiv-Search', 'Calculator-App', 'Python-Interpreter'))
      
      if options == 'Chroma-DB':
        st.markdown("""
        ## Chroma-DB
        Chroma HTTP.Client object class can be used to retrieve documents with metadata based on a corresponding query embedding
        """)
      elif options == 'Web-Search':
        st.markdown("""
        ## Web-Search
        A module which can search the web and just return the results
        """)
      elif options == 'arXiv-Search':
        st.markdown("""
        ## arXiv-Search
        A module which can search arXiv's research repository with abstracts, papers, and authors.
        """)
      elif options == 'Calculator-App':
        st.markdown("""
        ## Calculator-App
        A module to which can you send in a formula in the form of a string
        """)
      elif options == 'Python-Interpreter':
        st.markdown("""
        ## Python-Interpreter
        A module to which can you send in code as a string with delimiters, and get output back
        """)