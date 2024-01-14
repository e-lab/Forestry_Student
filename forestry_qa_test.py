import pandas_functions as BCB 
import pandas as pd 
import time
import openai


chatbot = BCB.ChatBot()
file_path = "/Users/viktorciroski/Documents/Github/Forestry_Student/TreeHugger_Exam_ans.csv"
print("test")
df = pd.read_csv(file_path)


results_dict = {"Question":[], "Pred":[], "GTruth":[], "GPT_Eval":[]}

for i in range(len(df['Question'])):
    try:
        question = df['Question'].iloc[i]
        answer = df['Answer'].iloc[i]
        
        response_chatbot = chatbot.agent({"input": f"{question}", "chat_history":[]})
        
        

        # Create a prompt to ask GPT if the answers are similar
        prompt = f"Are the following two answers saying the same thing?\n\nPredicted Answer: {response_chatbot['output']}\nGround Truth Answer: {answer}\n\nAnswer:"

        # Make an API call for evaluation
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=50,  # Adjust as needed
            n=1,  # Number of completions
            stop=None,
        )

        # Extract the GPT's response
        gpt_response = response['choices'][0]['text'].strip().lower()

        print(question)
        print(answer)
        print(response_chatbot['output'])
        print(f"GPT EVAL {gpt_response}")

        results_dict['Question'].append(question)
        results_dict['Pred'].append(response_chatbot['output'])
        results_dict['GTruth'].append(answer)
        results_dict['GPT_Eval'].append(gpt_response)
    except Exception as e:
        print("__________")
        print(e)
        print("Time limit error sleeping for 2 minutes")
        print("...")
        print("__________")
        time.sleep(120) 
    pd.DataFrame(results_dict).to_csv("Test_Results/results_test_1-6-24.csv", index=False)

    time.sleep(90) #To avoid OpenAI RateLimit Error


    #break
pd.DataFrame(results_dict).to_csv("Test_Results/results_test_1-6-24.csv", index=False)