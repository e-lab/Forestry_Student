from transformers import AutoTokenizer, LlamaTokenizer, pipeline
import transformers
import torch
import numpy as np
from numpy import pad
from matrix import similarity as similarity2

model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)


pipeline1 = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    # device_map="auto",  # TODO: uncomment on linux
    #device="cuda:0",
)


sequences = pipeline1(
    "In 2025, the aliens started to play chess with\n",
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)


for seq in sequences:
    print(f"Result: {seq['generated_text']}")
print(sequences)

del pipeline1

pipeline2 = transformers.pipeline(
    "feature-extraction", model=model, torch_dtype=torch.float32#, device="cuda:0"
)

embeds = pipeline2(
    [
        "I like to eat apples",
        "Wait hold on I did the leetcode daily so +5 mins eta. Also there’s 4 hands suddenly at the end so I’m working 2x slower. However, I’m almost done.",
    ],
    tokenizer=tokenizer,
    eos_token_id=tokenizer.eos_token_id,
)

del pipeline2

# print(embeds)
# print(len(embeds))

# cosine similarity of embeddings against themselves and other embeddings(or strings)
# Documents on how to get embeddings from llama2 (markdown on embedding)

from sklearn.metrics.pairwise import cosine_similarity

# for i, embed in enumerate(embeds):
#     print(f"Shape of embed[{i}]: {np.array(embed).shape}")
#     # embed = np.array(embed).reshape(1, -1)
#     embed = np.array(embed)[0][-1]
#     print(embed, embed.shape)
#     print(embed.reshape(1, -1), embed.reshape(1, -1).shape)

# similarity = cosine_similarity(
#     np.array(embeds[0])[0][-1].reshape(1, -1), np.array(embeds[1])[0][-1].reshape(1, -1)
# )
# print("Sim for last: ", similarity[0][0])


# similarity = cosine_similarity(
#     np.array(embeds[0])[0][0].reshape(1, -1), np.array(embeds[1])[0][0].reshape(1, -1)
# )
# print("Sim for first: ", similarity[0][0])

if len(embeds[0][0]) <= 1 or len(embeds[1][0]) <= 1:
    print("Not enough tokens to compare")
    exit()

emb1 = np.array(embeds[0])[0][1:][np.newaxis, :]
emb2 = np.array(embeds[1])[0][1:][np.newaxis, :]

print(emb1.shape, emb2.shape)
shorter_emb = emb1 if emb1.shape[1] < emb2.shape[1] else emb2
longer_emb = emb2 if shorter_emb is emb1 else emb1
padding_width = ((0, 0), (0, longer_emb.shape[1] - shorter_emb.shape[1]), (0, 0))
padded_shorter_emb = np.pad(shorter_emb, padding_width, mode="constant")
assert padded_shorter_emb.shape == longer_emb.shape

similarity = cosine_similarity(
    longer_emb.reshape(1, -1), padded_shorter_emb.reshape(1, -1)
)
print("Cosine with padding:", similarity[0][0])

print("Cosine with transpose: ", similarity2(emb1[0], emb2[0]))
