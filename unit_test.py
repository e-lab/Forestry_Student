import app 
import backend_chatbot as BCB

import pandas as pd 

def pdf_reader_test(file_path):
    text = app.read_pdf(file_path)
    print(len(text))

def take_exam(file_path="Test_Results/test_exam.txt"):
    chatbot = BCB.ChatBot()
    database_name='courses.db'
    chat_history = []
    ans_dict = {"Question":[], "Answer":[]}
    with open(file_path, 'r') as file:
        # Read file line by line
        counter = 0
        for line in file:
            # Process each line here (e.g., print or manipulate data)
            user_input = str(line.strip())
            try:
                print(f"Question:\t{user_input}")
                chat_history = chatbot.process_question(str(user_input), database_name, chat_history)
                print("______")
                print(chat_history[0]['message'].split('LLM Response:')[0])
                ans_dict['Question'].append(user_input)
                ans_dict['Answer'].append(chat_history[0]['message'].split('LLM Response:')[0])
            except:
                ans_dict['Question'].append(user_input)
                ans_dict['Answer'].append("N/A")
            counter += 1
            #break
    #print(ans_dict)
    pd.DataFrame(ans_dict).to_csv("TreeHugger_Exam_ans.csv")
        

if __name__ == '__main__':
    #file = "HOT_Draft1-2.pdf"
    #pdf_reader_test(file)
    take_exam()