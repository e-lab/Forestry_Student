from langchain.llms import HuggingFacePipeline
import torch
from datasets import load_dataset
from transformers import pipeline, GenerationConfig
from huggingface_hub import HfApi
from langchain.agents import load_tools, initialize_agent, Tool
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import BaseTool
import os


from tqdm import tqdm
# Define a function to evaluate the LLM on the dataset with a 6-shot exact match
def evaluate_llm(llm, dataset, hit_at_max_count=6):
    correct_predictions = 0
    total_examples = len(dataset)
    progress_bar = tqdm(total=total_examples, desc="Evaluating", unit="example")

    for example in dataset:
        question = example["question"]
        context = example["context"]
        answer = example["answer"]
        
        print("\n\n______") 
        print(f"gtruth {answer}")
        #print(f"context {context}")
        print(f"question {question}")
        print("_________")
        # Generate the top 6 answers
        counter  = 0
        while counter < hit_at_max_count:
            counter += 1
            try:
                generated_answers = llm( {"input": f"{question}", "chat_history":[]})
                print("_________________")
                print(generated_answers)
                print("_________________")
            
                # Check if any of the generated answers exactly match the reference answer
                if any(a.lower() == answer.lower() for a in generated_answers):
                    correct_predictions += 1
                    break
            except Exception as e:
                print(e)

        # Update the progress bar
        progress_bar.update(1)
        #break  

    # Close the progress bar
    progress_bar.close()

    accuracy = correct_predictions / total_examples
    return accuracy



model_id = 'meta-llama/Llama-2-13b-chat-hf'

hf_token = "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"
save_path = "/depot/euge/etc/models/"
scratch_data_dir = "/scratch/gilbreth/vciroski/forestry/qa_ir/data/squadv2"
#os.environ['TRANSFORMERS_CACHE'] = scratch_data_dir
api = HfApi()
api.token = hf_token
data = load_dataset("hotpot_qa", 'fullwiki', split='validation')#, data_dir=scratch_data_dir)

from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

class WordLengthTool(BaseTool):
    name = "Word Length Tool"
    description = "Use this tool when you need to supporting documents for a querry"

    def _run(self, word: str):
        return wikipedia.run(str(word))#str(len(word))

    def _arun(self, radius: int):
        raise NotImplementedError("This tool does not support async")
    
tools = [WordLengthTool()]
    
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up the generation config
gen_config = GenerationConfig.from_pretrained(
    model_id,
    token=hf_token,
    #cache_dir=scratch_data_dir
)
gen_config.max_new_tokens = 4096
gen_config.temperature = 1e-10

# Set up the Hugging Face pipeline for text generation
pipe = pipeline(
    task="text-generation",
    model=model_id,
    return_full_text=True,
    generation_config=gen_config,
    device_map=device,  # Use the determined device
    repetition_penalty=1.1
)

# Move the model to GPU if available
pipe.model.to(device)
llm = HuggingFacePipeline(pipeline=pipe)

#tools = load_tools(tools, llm=llm)
agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    early_stopping_method="generate",
    handle_parsing_errors=True, 
    max_iterations=3
    )

print("Langchian Base Prompt")
print(agent.agent.llm_chain.prompt)
# Special llama2 tokens that it was trained on.
B_INST, E_INST = "[INST]", "[/INST]" # Begin and end an instruction
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n" #Begin and end system prompt

# Define a new system prompt
sys_msg = B_SYS + """You are Assistant. Assistant is a expert JSON builder designed to assist with a wide range of tasks.

Assistant is able to respond to the User and use tools using JSON strings that contain "action" and "action_input" parameters.

All of Assistant's communication is performed using this JSON format. The assistant NEVER outputs anything other than a json object with an action and action_input fields!

Assistant can also use tools by responding to the user with tool use instructions in the same "action" and "action_input" JSON format. Tools available to Assistant are:

- "Word Length Tool": Useful for when you need to get the length of a word.
  - To use the Word Length Tool, Assistant should write like so:
    ```json
    {{"action": "Word Length Tool",
      "action_input": "elephant"}}
    ```

Here is an example of a previous conversation between User and Assistant:
---
User: Hey how are you today?
Assistant: ```json
{{"action": "Final Answer",
 "action_input": "I'm good thanks, how are you?"}}
```
User: I'm great, what is the length of the word educate?
Assistant: ```json
{{"action": "Word Length Tool",
 "action_input": "educate"}}
```
User: 7
Assistant: ```json
{{"action": "Final Answer",
 "action_input": "It looks like the answer is 7!"}}
```
User: Thanks could you tell me what the length of "turtle" is?
Assistant: ```json
{{"action": "Word Length Tool",
 "action_input": "turtle"}}
```
User: 6
Assistant: ```json
{{"action": "Final Answer",
 "action_input": "It looks like the answer is 6!"}}
```
---
Notice that after Assistant uses a tool, User will give the output of that tool. Then this output can be returned as a final answer.
Assistant will only use the available tools and NEVER a tool not listed. If the User's question does not require the use of a tool, Assistant will use the "Final Answer" action to give a normal response.
""" + E_SYS

print(f"sys msg \n\n {sys_msg}")

new_prompt = agent.agent.create_prompt(
    system_message=sys_msg,
    tools=tools
)

agent.agent.llm_chain.prompt = new_prompt

print(f"Agent custom Prompt \n\t{agent.agent.llm_chain.prompt}\n\n")
accuracy = evaluate_llm(agent, data)
print("Llama Langchain Agent custom Prompt")
print(f"LLM Accuracy (6-shot exact match) on HotpotQA Validation Dataset: {accuracy * 100:.2f}%")
