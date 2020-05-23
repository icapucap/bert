from transformers import GPT2LMHeadModel,GPT2Tokenizer
import torch as pt

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2",pad_token_id=tokenizer.eos_token_id)
input_context = 'I like eating fruits and ice cream'
#greedy search
input_ids = tokenizer.encode(input_context,return_tensors='pt')
greedy_output = model.generate(input_ids=input_ids,max_length=20)

# print(len(greedy_output))
#print(tokenizer.decode(greedy_output[0],skip_special_tokens=True))


#beam search
beam_output = model.generate(input_ids=input_ids,max_length=20,
num_beams=3,early_stopping=True)

# print(tokenizer.decode(beam_output[0],skip_special_tokens=True))
beam_output = model.generate(input_ids=input_ids,max_length=20,
num_beams=3,no_repeat_ngram_size=2,early_stopping=True)

# print(tokenizer.decode(beam_output[0],skip_special_tokens=True))

beam_outputs = model.generate(
    input_ids, 
    max_length=20, 
    num_beams=3, 
    no_repeat_ngram_size=2, 
    num_return_sequences=3, 
    early_stopping=True
)

# now we have 3 output sequences
# print("Output:\n" + 100 * '-')
# for i, beam_output in enumerate(beam_outputs):
#     print("{}: {}".format(i, tokenizer.decode(beam_output, skip_special_tokens=True)))

sample_output = model.generate(
    input_ids, 
    do_sample=True, 
    max_length=30, 
    top_k=0,
    temperature = 0.6
)

# print("Output:\n" + 100 * '-')
# print(tokenizer.decode(sample_output[0], skip_special_tokens=True))

sample_output = model.generate(
    input_ids, 
    do_sample=True,
    no_repeat_ngrams_size=3, 
    max_length=30, 
    top_k=0,
    top_p = 0.94
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))