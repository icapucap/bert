This is a language model based on gpt2 for hindi language.Given a starting hindi phrase and model can generate hindi text.<br>
lang-model-hin.py trains a pretrained gpt2 model on a hindi text corpus, for tokenizing a Bert-pretrained tokenizer is used.


# to run
```
export TRAIN_FILE=/path/to/dataset/wiki.train.raw
export TEST_FILE=/path/to/dataset/wiki.test.raw

python lang-modelling.py \
    --output_dir=output \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE

```
