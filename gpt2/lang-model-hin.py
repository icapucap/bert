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
    output_dir="/home/shidhu/gpt2/output",
    overwrite_output_dir=True,
    num_train_epochs=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True,
)

trainer.train()
trainer.save_model("/home/shidhu/gpt2/output")