from pathlib import Path
from transformers import BertTokenizer,GPT2LMHeadModel
from tokenizers import ByteLevelBPETokenizer


# Initialize a tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")


from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing




from transformers import GPT2Config

config = GPT2Config(vocab_size=tokenizer.vocab_size)


model = GPT2LMHeadModel(config=config)
from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="/home/shidhu/gpt2/hin-only.txt",
    block_size=8,
)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="/content/output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    logging_dir='/content/output',save_steps=10_000,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True

)

trainer.train()
trainer.save_model("/content/output")

input_context = '	बहुत समय से देखा नहीं '
#greedy search
input_ids = tokenizer.encode(input_context,return_tensors='pt')
# greedy_output = model.generate(input_ids=input_ids,max_length=50)
beam_outputs = model.generate(
    input_ids, 
    max_length=50, 
    num_beams=3, 
    no_repeat_ngram_size=2, 
    num_return_sequences=3, 
    early_stopping=True
)
# print(len(greedy_output))
print(tokenizer.decode(beam_outputs[2],skip_special_tokens=True))
