import torch
from transformers import BertConfig, BertModel, BertForMaskedLM, BertTokenizer,BertLMHeadModel
from transformers import DistilBertConfig
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertForMaskedLM
from transformers.generation_utils import GenerationMixin
from transformers import GPT2Model, GPT2Config
import pytorch_lightning as pl
from easydict import EasyDict as ED
from pathlib import Path
from transformers import WEIGHTS_NAME, CONFIG_NAME
import numpy as np
from config import Config
from torch.utils.data import Dataset, DataLoader
import inspect
import logging
import os
from typing import Callable, Dict, Iterable, List, Optional, Tuple


class TranslationModel(pl.LightningModule):

    def __init__(self, config,src_tokenizer,tgt_tokenizer,encoder,decoder):

        super().__init__() 

        #Creating encoder and decoder with their respective embeddings.
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.src_tokenizers = src_tokenizer
        self.tgt_tokenizers = tgt_tokenizer
        

    def forward(self, encoder_input_ids, decoder_input_ids):

        

        encoder_hidden_states = self.encoder(encoder_input_ids)[0]
        loss, logits = self.decoder(input_ids=decoder_input_ids,
                                    encoder_hidden_states=encoder_hidden_states, 
                                    labels=decoder_input_ids)
        
        return loss, logits
    
    
    
    
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
    
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        source = batch[0].to(device)
        target = batch[1].to(device)

        loss, logits = self.forward(source,target)
        logits = logits.detach().cpu().numpy()
        label_ids = target.to('cpu').numpy()
        
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}
    
    def validation_step(self, batch, batch_no):
    
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        source = batch[0].to(device)
        target = batch[1].to(device)

        loss, logits = self.forward(source,target)
        logits = logits.detach().cpu().numpy()
        label_ids = target.to('cpu').numpy()
        
        eval_accuracy = torch.from_numpy(np.asarray(self.flat_accuracy(logits, label_ids)))
       
        return {'eval_acc': eval_accuracy,'eval_loss': loss}


    def validation_epoch_end(self,outputs):
        val_loss = torch.stack([x['eval_loss'] for x in outputs]).mean()
        val_acc = torch.stack([x['eval_acc'] for x in outputs]).mean()
        log = {'avg_val_loss':val_loss}
        tqdm_dict = {'val_loss': val_loss.item(), 'val_acc': val_acc.item()}
        return {'log':log,'val_loss':log,'progress_bar': tqdm_dict }
    
    def test_step(self,batch,batch_no):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        source = batch[0].to(device)
        # print(source)
        
        output = self.generate(source)
        # print(output)
        print(self.tgt_tokenizers.decode(output[0]))

    # def test_step(self, batch, batch_no):
    #     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
    #     source = batch[0].to(device)
    #     target = batch[1].to(device)

    #     loss, logits = self.forward(source,target)
        
    #     logits = logits.detach().cpu().numpy()
    #     label_ids = target.to('cpu').numpy()
        
    #     test_accuracy = torch.from_numpy(np.asarray(self.flat_accuracy(logits, label_ids)))
       
    #     return {'test_acc': test_accuracy}

    # def test_epoch_end(self, outputs):

    #     avg_test_acc = torch.stack([x['test_acc'] for x in outputs]).mean()

    #     tensorboard_logs = {'avg_test_acc': avg_test_acc}
    #     return {'avg_test_acc': avg_test_acc, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}
    

   
   
    # def validation_end(self, outputs):
    #     val_loss_mean = 0
    #     val_acc_mean = 0
    #     for output in outputs:
    #         val_loss_mean += output['eval_loss']
    #         val_acc_mean += output['eval_acc']

    #     val_loss_mean /= len(outputs)
    #     val_acc_mean /= len(outputs)
    #     tqdm_dict = {'val_loss': val_loss_mean.item(), 'val_acc': val_acc_mean.item()}

    #     # show val_loss and val_acc in progress bar but only log val_loss
    #     results = {
    #         'progress_bar': tqdm_dict,
    #         'log': {'val_loss': val_loss_mean.item()}
    #     }
    #     return results

    def prepare_data(self):
        from data import split_data
        split_data('/content/itr/hin.txt', '/content/itr/')
        # split_data('/home/shidhu/itr/itr/hin.txt', '/home/shidhu/itr/itr/')

    
    def train_dataloader(self):
        from data import IndicDataset, PadSequence
        pad_sequence = PadSequence(self.src_tokenizers.pad_token_id, self.tgt_tokenizers.pad_token_id)

        return DataLoader(IndicDataset(self.src_tokenizers, self.tgt_tokenizers, self.config.data, True, False), 
                                batch_size=self.config.batch_size, 
                                shuffle=False, 
                                collate_fn=pad_sequence)
    def val_dataloader(self):
        from data import IndicDataset, PadSequence
        pad_sequence = PadSequence(self.src_tokenizers.pad_token_id, self.tgt_tokenizers.pad_token_id)

        return DataLoader(IndicDataset(self.src_tokenizers, self.tgt_tokenizers, self.config.data, False, False), 
                           batch_size=self.config.eval_size, 
                           shuffle=False, 
                           collate_fn=pad_sequence)

    def test_dataloader(self):
        from data import IndicDataset, PadSequence
        pad_sequence = PadSequence(self.src_tokenizers.pad_token_id, self.tgt_tokenizers.pad_token_id)

        return DataLoader(IndicDataset(self.src_tokenizers, self.tgt_tokenizers, self.config.data, False,True), 
                           batch_size=1, 
                           shuffle=False, 
                           collate_fn=pad_sequence)
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        return optimizer

    def configure_schedulers(self):
        optimizer = self.configure_optimizers()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=self.config.lr, eps=1e-08)
        return scheduler

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}


    def generate(self,input_ids,max_length=20):
        batch_size=input_ids.shape[0]
        max_length=max_length
        bos_token_id = self.tgt_tokenizers.bos_token_id
        pad_token_id = self.tgt_tokenizers.pad_token_id
        eos_token_id = self.tgt_tokenizers.eos_token_id
        num_beams=1
    
        attention_mask = input_ids.ne(pad_token_id).long()

        effective_batch_size = batch_size
        effective_batch_mult = 1
        decoder_start_token_id = bos_token_id

        # encoder_outputs: tuple = self.encoder(input_ids, attention_mask=attention_mask)
        #dummy input_ids for decoder
        input_ids_dummy = torch.full(
                (effective_batch_size * num_beams, 1),
                decoder_start_token_id,
                dtype=torch.long,
                device=next(self.parameters()).device,
            )
        cur_len = 1

        # assert (
        #     batch_size == encoder_outputs[0].shape[0]
        # ), f"expected encoder_outputs[0] to have 1st dimension bs={batch_size}, got {encoder_outputs[0].shape[0]} "

        # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
        # expanded_batch_idxs = (
        #     torch.arange(batch_size)
        #     .view(-1, 1)
        #     .repeat(1, num_beams * effective_batch_mult)
        #     .view(-1)
        #     .to(input_ids.device)
        # )
        # # expand encoder_outputs
        # encoder_outputs = (encoder_outputs[0].index_select(0, expanded_batch_idxs), *encoder_outputs[1:])


        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)
        
        # past = (encoder_outputs, None)
        
        while cur_len<max_length:
            # model_inputs = self.prepare_inputs_for_generation(
            #     input_ids,past=past,attention_mask=attention_mask
            # )

            outputs = self.forward(input_ids,input_ids_dummy)
            next_token_logits = outputs[:,-1,:]
            
                # Greedy decoding
            next_token = torch.argmax(next_token_logits, dim=-1)

            # update generations and finished sentences
            if eos_token_id is not None:
                # pad finished sentences if eos_token_id exist
                tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token

            # add token and increase length by one
            input_ids_dummy = torch.cat([input_ids_dummy, tokens_to_add.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            if eos_token_id is not None:
                eos_in_sents = tokens_to_add == eos_token_id
                # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
                # unfinished_sents is set to zero if eos in sentence
                unfinished_sents.mul_((~eos_in_sents).long())

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

            # extend attention_mask for new generated input if only decoder
            
        # if there are different sentences lengths in the batch, some batches have to be padded
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`Pad_token_id` has to be defined if batches have different lengths"
            # finished sents are filled with pad_token
            decoded = input_ids_dummy.new(batch_size, sent_lengths.max().item()).fill_(pad_token_id)
        else:
            decoded = input_ids_dummy

        for hypo_idx, hypo in enumerate(input_ids_dummy):
            decoded[hypo_idx, : sent_lengths[hypo_idx]] = hypo[: sent_lengths[hypo_idx]]

        return decoded

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
    
    
    
    encoder = BertModel(encoder_config)

    encoder_embeddings = torch.nn.Embedding(src_tokenizer.vocab_size, config.hidden_size, padding_idx=src_tokenizer.pad_token_id)
    decoder_embeddings = torch.nn.Embedding(tgt_tokenizer.vocab_size, config.hidden_size, padding_idx=tgt_tokenizer.pad_token_id)

    
    decoder = BertLMHeadModel(decoder_config)
    encoder.set_input_embeddings(encoder_embeddings.cuda())
    decoder.set_input_embeddings(decoder_embeddings.cuda())

    # #Create encoder and decoder embedding layers.
    # encoder_embeddings = torch.nn.Embedding(src_tokenizer.vocab_size, config.hidden_size, padding_idx=src_tokenizer.pad_token_id)
    # decoder_embeddings = torch.nn.Embedding(tgt_tokenizer.vocab_size, config.hidden_size, padding_idx=tgt_tokenizer.pad_token_id)

    # encoder = BertModel(encoder_config)
    # encoder.set_input_embeddings(encoder_embeddings.cuda())
    
    # decoder = BertForMaskedLM(decoder_config)
    # decoder.set_input_embeddings(decoder_embeddings.cuda())
    model = TranslationModel(config,src_tokenizer,tgt_tokenizer,encoder,decoder)
    model.cuda()

    tokenizers = ED({'src':src_tokenizer,'tgt':tgt_tokenizer})

    
    return model,tokenizers
# from config import replace, preEnc, preEncDec
# rconf = preEncDec
# model = TranslationModel(rconf)
# print("Model's state_dict:")
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())










