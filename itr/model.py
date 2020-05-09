import torch
from transformers import BertConfig, BertModel, BertForMaskedLM, BertTokenizer
import pytorch_lightning as pl
from easydict import EasyDict as ED
from pathlib import Path
from transformers import WEIGHTS_NAME, CONFIG_NAME
import numpy as np
from config import Config
from torch.utils.data import Dataset, DataLoader

class TranslationModel(pl.LightningModule):

    def __init__(self, encoder, decoder, tokenizers, config):

        super().__init__() 

        #Creating encoder and decoder with their respective embeddings.
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizers = tokenizers
        self.config = config
        

    def forward(self, encoder_input_ids, decoder_input_ids):

        

        encoder_hidden_states = self.encoder(encoder_input_ids)[0]
        loss, logits = self.decoder(decoder_input_ids,
                                    encoder_hidden_states=encoder_hidden_states, 
                                    masked_lm_labels=decoder_input_ids)

        return loss, logits
    
    
    
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #device = torch.device("cpu")

    # def save_model(self, output_dir):

    #     output_dir = Path(output_dir)
    #     # Step 1: Save a model, configuration and vocabulary that you have fine-tuned

    #     # If we have a distributed model, save only the encapsulated model
    #     # (it was wrapped in PyTorch DistributedDataParallel or DataParallel)
    #     model_to_save = self.module if hasattr(self, 'module') else self

    #     # If we save using the predefined names, we can load using `from_pretrained`
    #     output_model_file = output_dir / WEIGHTS_NAME
    #     output_config_file = output_dir / CONFIG_NAME

    #     torch.save(model_to_save.state_dict(), output_model_file)
    #     model_to_save.config.to_json_file(output_config_file)
    #     #src_tokenizer.save_vocabulary(output_dir)

    # def load_model(self):
    #     pass

    # Function to calculate the accuracy of our predictions vs labels
    def flat_accuracy(self,preds, labels):

        pred_flat = np.argmax(preds, axis=2).flatten()
        labels_flat = labels.flatten()
        #print (f'preds: {pred_flat}')
        #print (f'labels: {labels_flat}')

        return np.sum(np.equal(pred_flat, labels_flat)) / len(labels_flat)


    # def save(self, tokenizers, output_dirs):
    #     # from train_util import save_model

    #     save_model(self.encoder, output_dirs.encoder)
    #     save_model(self.decoder, output_dirs.decoder)

    

    def training_step(self, batch, batch_no):
    
        source = batch[0].to(device)
        target = batch[1].to(device)

        loss, logits = TranslationModel(source,target)
        logits = logits.detach().cpu().numpy()
        label_ids = target.to('cpu').numpy()
        
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}
    
    def validation_step(self, batch, batch_no):
        source = batch[0].to(device)
        target = batch[1].to(device)

        loss, logits = TranslationModel(source,target)
        logits = logits.detach().cpu().numpy()
        label_ids = target.to('cpu').numpy()
        
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        eval_loss += loss
        nb_eval_steps += 1

        return {'eval_acc': eval_accuracy,'eval_loss': eval_loss}

    def validation_epoch_end(self,outputs):

        avg_loss = torch.stack([x['eval_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['eval_acc'] for x in outputs]).mean()
        tensorboard_logs = {'eval_loss': avg_loss, 'eval_acc': avg_acc}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def prepare_data(self):
        from data import split_data
        split_data('itr/hin.txt', 'itr/')

    
    def train_dataloader(self):
        from data import IndicDataset, PadSequence
        pad_sequence = PadSequence(self.tokenizers.src.pad_token_id, self.tokenizers.tgt.pad_token_id)

        return DataLoader(IndicDataset(self.tokenizers.src, self.tokenizers.tgt, self.config.data, True), 
                                batch_size=self.config.batch_size, 
                                shuffle=False, 
                                collate_fn=pad_sequence)
    def val_dataloader(self):
        from data import IndicDataset, PadSequence
        pad_sequence = PadSequence(self.tokenizers.src.pad_token_id, self.tokenizers.tgt.pad_token_id)

        return DataLoader(IndicDataset(self.tokenizers.src, self.tokenizers.tgt, self.config.data, False), 
                           batch_size=self.config.eval_size, 
                           shuffle=False, 
                           collate_fn=pad_sequence)
    
    def configure_optimizers(self):
        from config import Config
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        return optimizer

    def configure_schedulers(self):
        from config import Config
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader), eta_min=self.config.lr)
        return scheduler


def build_model(config):
    
    src_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    tgt_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    tgt_tokenizer.bos_token = '<s>'
    tgt_tokenizer.eos_token = '</s>'
    
    #hidden_size and intermediate_size are both wrt all the attention heads. 
    #Should be divisible by num_attention_heads
    encoder_config = BertConfig(vocab_size=src_tokenizer.vocab_size,
                                hidden_size=config.hidden_size,
                                num_hidden_layers=config.num_hidden_layers,
                                num_attention_heads=config.num_attention_heads,
                                intermediate_size=config.intermediate_size,
                                hidden_act=config.hidden_act,
                                hidden_dropout_prob=config.dropout_prob,
                                attention_probs_dropout_prob=config.dropout_prob,
                                max_position_embeddings=512,
                                type_vocab_size=2,
                                initializer_range=0.02,
                                layer_norm_eps=1e-12)

    decoder_config = BertConfig(vocab_size=tgt_tokenizer.vocab_size,
                                hidden_size=config.hidden_size,
                                num_hidden_layers=config.num_hidden_layers,
                                num_attention_heads=config.num_attention_heads,
                                intermediate_size=config.intermediate_size,
                                hidden_act=config.hidden_act,
                                hidden_dropout_prob=config.dropout_prob,
                                attention_probs_dropout_prob=config.dropout_prob,
                                max_position_embeddings=512,
                                type_vocab_size=2,
                                initializer_range=0.02,
                                layer_norm_eps=1e-12,
                                is_decoder=True)

    #Create encoder and decoder embedding layers.
    encoder_embeddings = torch.nn.Embedding(src_tokenizer.vocab_size, config.hidden_size, padding_idx=src_tokenizer.pad_token_id)
    decoder_embeddings = torch.nn.Embedding(tgt_tokenizer.vocab_size, config.hidden_size, padding_idx=tgt_tokenizer.pad_token_id)

    encoder = BertModel(encoder_config)
    encoder.set_input_embeddings(encoder_embeddings.cuda())
    
    decoder = BertForMaskedLM(decoder_config)
    decoder.set_input_embeddings(decoder_embeddings.cuda())

    tokenizers = ED({'src': src_tokenizer, 'tgt': tgt_tokenizer})

    model = TranslationModel(encoder, decoder, tokenizers, config)
    model.cuda()


    
    return model , tokenizers










