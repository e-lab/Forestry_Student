import streamlit as st
import base64, os 
import time 
import io
import glob

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

    disabled = False 
    if st.session_state['api_key']:
      disabled = True
    key = st.sidebar.text_input('', placeholder ='Input your OpenAI API Key: ', type='password', label_visibility='hidden', key='api_key_input', disabled=disabled)
    
    if key: 
      st.session_state['api_key'] = key
      st.sidebar.success('API Key Successfully Added!')
    st.sidebar.divider() 

    self._show_tools()
  
    if st.session_state['api_key']: 
      return 1
    else: 
      return 0 

  def _upload_widget(self): 

    upload_expander = st.sidebar.expander("File Uploader", expanded=True)
    with upload_expander: 
      pdf_docs = st.file_uploader(label='Select Files to Upload', accept_multiple_files=True, type=['pdf', 'txt', 'csv'])
      if st.button('Start Upload'): 

        if len(self.pipeline.document_handler) > 5: 
          st.error('Document limit reached for this demo app!')
          return
          
        for pdf in pdf_docs:

          progress_text = 'Checking File...'
          my_bar = st.progress(0, text=progress_text)
          percent_complete = 0

          percent_complete += 50 
          my_bar.progress(percent_complete, text=progress_text)
          progress_text = 'Processing File...'

          if pdf.type == 'application/pdf':
            with open(f"{os.environ['TMP']}/{pdf.name}", 'wb') as f: 
              f.write(pdf.read())
          else: 
            with open(f"{os.environ['TMP']}/{pdf.name}", 'w') as f: 
              f.write(pdf.read())
            
          percent_complete += 50
          my_bar.progress(percent_complete, text="Finalizing...")
          st.success(f'File Successfully Processed!')

          my_bar.empty()
          
          del my_bar
    
  def _delete_widget(self): 

    del_expander = st.sidebar.expander("File Deleter", expanded=True)
    with del_expander: 
      rag  = st.multiselect('Documents', 
          [fil.replace(f"{os.environ['TMP']}", '') for fil in glob.glob(f"{os.environ['TMP']}/*.pdf")]+[fil.replace(f"{os.environ['TMP']}", '') for fil in glob.glob(f"{os.environ['TMP']}/*.txt")], 
          placeholder="Select Document(s) to remove", key="rag_del")
      
      csv  = st.multiselect('Data Files', 
          [fil.replace(f"{os.environ['TMP']}", '') for fil in glob.glob(f"{os.environ['TMP']}/*.csv")], 
          placeholder="Select CSV(s) to remove", key="csv_del")

      if st.button('Delete Files'): 
        files = rag + csv
        for fil in files: 
          os.remove(f"{os.environ['TMP']}"+fil)
        st.success('Files Successfully Deleted!')

            
  def _show_tools(self):
      tools = st.sidebar.expander(label="Tools", expanded=True)
      with tools:
          options = st.selectbox(
              'List of available tools: ',
              ('InContext QA', 'Python Interpreter', 'ArXiv Search', 'Calculator', 'Web Search')
          )

          if options == 'InContext QA':
              st.markdown("""
              A tool that leverages a vector store for retrieving documents with metadata based on query embeddings. Ideal for contextual inquiries and deep dives into specific topics.
              """)
          elif options == 'Python Interpreter':
              st.markdown("""
              A flexible module where you can send Python code as a string, execute it, and receive the output. Supports a wide range of Python functionality for dynamic computing.
              """)
          elif options == 'ArXiv Search':
              st.markdown("""
              Directly search arXiv's vast repository of research papers, abstracts, and authors to find academic works relevant to your query. Essential for researchers and scholars.
              """)
          elif options == 'Calculator':
              st.markdown("""
              Send mathematical formulas as strings to this module to calculate and return results instantly. Useful for quick calculations without leaving the application.
              """)
          elif options == 'Web Search':
              st.markdown("""
              Utilizes a combination of LLM and a public vector store to perform web searches, returning relevant results. Streamlines information retrieval directly within the application.
              """)
          elif options == 'Data Agent':
              st.markdown("""
              Allows people to chat with user's data (<10MB/file) and make visualizations with personal data. 
              """)


