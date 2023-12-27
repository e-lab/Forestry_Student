from flask import Flask, render_template, request, jsonify
from langchain.memory import ChatMessageHistory
import os
from flask import request
import os
import fitz  # PyMuPDF

import textwrap


#import backend_chatbot as BCB
import pandas_functions as BCB 

current_directory = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_url_path='/static')
app = Flask(__name__, template_folder='templates')
app.secret_key = 'your_secret_key'
app = Flask(__name__)
global chat_history
global file_path_msg
file_path_msg = None
history = ChatMessageHistory()
message = f"Hello my name is FIRST (Forest Intellect Research & \nTechnology System), how can i help you today?"
history.add_ai_message(message)
chat_history = [{'user': False, 'message': message}]
chatbot = BCB.ChatBot()
database_name='courses.db'

# Your existing code here...

@app.route('/')
def index():
    return render_template('index.html', chat_history=chat_history)

def save_file(file):
    # Process the file as needed (save to disk, etc.)
    # For example, save the file to the 'uploads' folder
    uploads_folder = 'uploads'
    if not os.path.exists(uploads_folder):
        os.makedirs(uploads_folder)

    # Ensure the file name is unique to avoid overwriting directories
    file_name = file.filename
    while os.path.exists(os.path.join(uploads_folder, file_name)):
        # If a file with the same name exists, append a number to make it unique
        name, extension = os.path.splitext(file_name)
        file_name = f"{name}_1{extension}"

    # Save the file to the 'uploads' folder
    file_path = os.path.join(uploads_folder, file_name)
    file.save(file_path)
    return file_path

@app.route('/upload_material', methods=['POST'])
def upload_material():
    """TEST FUNCTIONALITY OF EMBEEDING DATA THROUGHT 
    Does the class and topic name come in 
    Does the file fully get converted to text 
    will it be split to chunks 
    can we save in the vectore store 
    """
    
    # Check if the request contains a file part
    if 'fileInput' not in request.files:
        return "No file provided", 400

    file = request.files['fileInput']

    # Check if the file name is empty
    if file.filename == '':
        return "No file selected", 400

    class_name = request.form.get('className').lower()
    topic_name = request.form.get('topics').lower()

    file_path  = save_file(file)

    # Read the content based on file type
    if file.filename.endswith('.txt'):
        with open(file_path, 'r') as txt_file:
            file_contents = txt_file.read()
    elif file.filename.endswith('.pdf'):
        file_contents = read_pdf(file_path)
    else:
        return "Unsupported file format", 400

    # Perform any additional processing
    # Assuming chatbot and add_embeddings functions are correctly defined
    chatbot.add_embeddings(file_contents, file.filename, class_name, topic_name, database_name)

    try:
        os.remove(file_path)
        print(f"File {file_path} deleted successfully.")
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Display uploaded file information
    #return f"File '{file.filename}' uploaded successfully. Class: {class_name}, Topics: {topic_name}"
    return render_template('index.html', chat_history=chat_history)


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




def format_string(input_string, n):
        formatted_string = textwrap.wrap(input_string, width=n)
        return '\n'.join(formatted_string)

@app.route('/process_input', methods=['POST'])
def process_input():
    global file_path_msg
    print("Chat Box")
    user_input = request.form.get('user_input')
    print(user_input)
    file = request.files['fileInputMessage'] if 'fileInputMessage' in request.files else None
    print(file)
    if file is not None:
        #print(f"file name: {file}")
        file_path_new  = save_file(file)
        if file_path_new.startswith("_1"):
            pass 
        else:
            """STILL NEED WAY TO KEEP WORKING FILE STATIC TILL USER CHANGES BUT NOT NEEDED"""
            #print(file_path_new)x
            if file_path_new == file_path_msg:
                pass 
            else:
                file_path_msg = file_path_new
    print(file_path_msg)
    global chat_history
    # Add user input to the chat history
    print(user_input)
    chat_history.append({'user': True, 'message': user_input})
    print("Added to chat history")
    history.add_user_message(user_input)
    #csv_path = "/Users/viktorciroski/Documents/Github/Forestry_Student/indiana_trees_remeasured.csv"
  
    response = chatbot.agent({"input": f"{user_input} {file_path_msg}", "chat_history":[]})#chat_history})#Error with chat history formatting... hasn't been a problem till 12/27 2:30pm. Not sure what's happened
    output = format_string(response['output'], n=60)
    print(output)
    chat_history.append({'user': False, 'message': output})
    history.add_ai_message(output)
    print("\n\n\n\n\n\n\n")
    print(history.messages)
    print("\n\n\n\n\n\n\n")
    #chat_history.append({'user': False, 'message': history.messages[0]})
    

    return render_template('index.html', chat_history=chat_history)



if __name__ == '__main__':
    app.run(debug=True)
