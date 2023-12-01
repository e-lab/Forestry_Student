from flask import Flask, render_template, request, jsonify
import os
from flask import request
import os
import fitz  # PyMuPDF

import backend_chatbot as BCB

current_directory = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_url_path='/static')
app = Flask(__name__, template_folder='templates')
app.secret_key = 'your_secret_key'
app = Flask(__name__)
chat_history = []
chatbot = BCB.ChatBot()
database_name='courses.db'

# Your existing code here...

@app.route('/')
def index():
    return render_template('index.html')



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

    # Process the file as needed (save to disk, etc.)
    # For example, save the file to the 'uploads' folder
    uploads_folder = 'uploads'
    if not os.path.exists(uploads_folder):
        os.makedirs(uploads_folder)

    # Save the file to the 'uploads' folder
    file_path = os.path.join(uploads_folder, file.filename)
    file.save(file_path)

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


@app.route('/process_input', methods=['POST'])
def process_input():
    print("Chat Box")
    global chat_history
    user_input = request.form.get('user_input')
    
    # Add user input to the chat history
    chat_history.append({'user': True, 'message': user_input})
    chat_history = chatbot.process_question(user_input, database_name, chat_history)

    return render_template('index.html', chat_history=chat_history)



if __name__ == '__main__':
    app.run(debug=True)
