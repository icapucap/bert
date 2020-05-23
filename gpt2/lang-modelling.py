import torch
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    '''
    arguments for model,config,tokenizer
    '''
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "model checkpoints for wt, leave to train from scratch"
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help":"choose from list:"+",".join(MODEL_TYPES)},
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

@dataclass
class DataTrainingArguments:
    '''
    input for the model 
    '''
    train_data_file: Optional[str] = field(
        default=None,
        metadata={
            "help":"input training data .txt file"
        },
    )
    eval_data_file: Optional[str] = field(
        default=None,metadata={"help":"eval data .txt file"},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )

def get_dataset(args: DataTrainingArguments,tokenizer: PreTrainedTokenizer,evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    return TextDataset(
        tokenizer=tokenizer,file_path=file_path,block_size=args.block_size, overwrite_cache=args.overwrite_cache
    )

def main():
    parser = HfArgumentParser((ModelArguments,DataTrainingArguments,TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
        "cannot do eval without eval data"
    )

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)


    if model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path,cache_dir=model_args.cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,cache_dir=model_args.cache_dir)
        model = AutoModelWithLMHead.from_pretrained(model_args.model_name_or_path,from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("instantiating new config instance")
        logger.info("Training new model from scratch")
        model = AutoModelWithLMHead.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    if config.model_type in ["bert", "roberta", "distilbert", "camembert"] and not data_args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
            "flag (masked language modeling)."
        )

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)


    train_dataset = get_dataset(data_args,tokenizer=tokenizer) if training_args.do_train else None
    eval_dataset = get_dataset(data_args,tokenizer=tokenizer,evaluate=True) if training_args.do_eval else None
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer
    )

    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=True,
    )

    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.train(model_path=model_path)
        trainer.save_model()

    results = {}
    if training_args.do_eval:
        logger.info("***eval***")
        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        result = {"perplexity":perplexity}

        output_eval_file = os.path.join(training_args.output_dir,"eval_results.txt")

        with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

    return results

if __name__=="__main__":
    main()
