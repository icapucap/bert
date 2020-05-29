import torch
from transformers import BertConfig, BertModel, BertForMaskedLM, BertTokenizer
from transformers import DistilBertConfig
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertForMaskedLM
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
        loss, logits = self.decoder(decoder_input_ids,
                                    encoder_hidden_states=encoder_hidden_states, 
                                    lm_labels=decoder_input_ids)

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
        input_id = batch[0]
        output = self.generate(input_id.to(device),max_length=20)
        print(self.tgt_tokenizers.decode(output[0],skip_special_tokens=True))


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
                           batch_size=self.config.eval_size, 
                           shuffle=False, 
                           collate_fn=pad_sequence)
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        return optimizer

    def configure_schedulers(self):
        optimizer = self.configure_optimizers()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.train_dataloader()), eta_min=self.config.lr)
        return scheduler

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_start_token_id: Optional[int] = None,
        **model_specific_kwargs
    ) -> torch.LongTensor:
        
        # We cannot generate if the model does not have a LM head
        if self.decoder.get_output_embeddings() is None:
            raise AttributeError(
                "You tried to generate sequences with a model that does not have a LM Head."
                "Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`, `XLMWithLMHeadModel`, `BartForConditionalGeneration` )"
            )

        max_length = max_length if max_length is not None else self.config.max_length
        min_length = min_length if min_length is not None else self.config.min_length
        bos_token_id = bos_token_id if bos_token_id is not None else self.tgt_tokenizers.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.tgt_tokenizers.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.tgt_tokenizers.eos_token_id
        
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else self.tgt_tokenizers.bos_token_id
        )

        if input_ids is not None:
            batch_size = input_ids.shape[0]  # overriden by the input batch_size
        else:
            batch_size = 1

        assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
        assert isinstance(min_length, int) and min_length >= 0, "`min_length` should be a positive integer."
        
        assert input_ids is not None or (
            isinstance(bos_token_id, int) and bos_token_id >= 0
        ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
        assert pad_token_id is None or (
            isinstance(pad_token_id, int) and (pad_token_id >= 0)
        ), "`pad_token_id` should be a positive integer."
        assert (eos_token_id is None) or (
            isinstance(eos_token_id, int) and (eos_token_id >= 0)
        ), "`eos_token_id` should be a positive integer."
        
        if input_ids is None:
            assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
                "you should either supply a context to complete as `input_ids` input "
                "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
            )
            input_ids = torch.full(
                (batch_size, 1), bos_token_id, dtype=torch.long, device=torch.device("cuda"),
            )
        else:
            assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."

        # not allow to duplicate outputs when greedy decoding
        
        # create attention mask if necessary
        # TODO (PVP): this should later be handled by the forward fn() in each model in the future see PR 3140
        if (attention_mask is None) and (pad_token_id is not None) and (pad_token_id in input_ids):
            attention_mask = input_ids.ne(pad_token_id).long()
        elif attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        # set pad_token_id to eos_token_id if not set. Important that this is done after
        # attention_mask is created
        if pad_token_id is None and eos_token_id is not None:
            logger.warning(
                "Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence".format(eos_token_id)
            )
            pad_token_id = eos_token_id

        
        vocab_size = self.tgt_tokenizers.vocab_size

        # set effective batch size and effective batch multiplier according to do_sample
        
        effective_batch_size = batch_size
        effective_batch_mult = 1

        
        # get encoder and store encoder outputs
        encoder = self.encoder

        encoder_outputs: tuple = encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Expand input ids if num_beams > 1 or num_return_sequences > 1

        
        # create empty decoder_input_ids
        input_ids = torch.full(
            (effective_batch_size * 1, 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=torch.device('cuda'),
        )
        cur_len = 1

        assert (
            batch_size == encoder_outputs[0].shape[0]
        ), f"expected encoder_outputs[0] to have 1st dimension bs={batch_size}, got {encoder_outputs[0].shape[0]} "

        # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
        expanded_batch_idxs = (
            torch.arange(batch_size)
            .view(-1, 1)
            .repeat(1, 1 * effective_batch_mult)
            .view(-1)
            .to(input_ids.device)
        )
        # expand encoder_outputs
        encoder_outputs = (encoder_outputs[0].index_select(0, expanded_batch_idxs), *encoder_outputs[1:])

        
        output = self._generate_no_beam_search(
                    input_ids,
                    cur_len=cur_len,
                    max_length=max_length,
                    min_length=min_length,
                    bos_token_id=bos_token_id,
                    pad_token_id=pad_token_id,
                    decoder_start_token_id=decoder_start_token_id,
                    eos_token_id=eos_token_id,
                    batch_size=effective_batch_size,
                    encoder_outputs=encoder_outputs,
                    attention_mask=attention_mask,
                    model_specific_kwargs=model_specific_kwargs,
                )

        return output
    
    def _generate_no_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        min_length,
        bos_token_id,
        pad_token_id,
        eos_token_id,
        decoder_start_token_id,
        batch_size,
        encoder_outputs,
        attention_mask,
        model_specific_kwargs,
    ):
        """ Generate sequences for each example without beam search (num_beams == 1).
            All returned sequence are generated independantly.
        """
        # length of generated sentences / unfinished sentences
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)

        past = encoder_outputs  # defined for encoder-decoder models, None for decoder-only models

        while cur_len < max_length:
            model_inputs = self.decoder.prepare_inputs_for_generation(
                input_ids, past=past, attention_mask=attention_mask, **model_specific_kwargs
            )

            outputs = self.decoder(**model_inputs)
            next_token_logits = outputs[0][:, -1, :]

           
            # set eos token prob to zero if min_length is not reached
            if eos_token_id is not None and cur_len < min_length:
                next_token_logits[:, eos_token_id] = -float("inf")

           
                # Greedy decoding
            next_token = torch.argmax(next_token_logits, dim=-1)

            # update generations and finished sentences
            if eos_token_id is not None:
                # pad finished sentences if eos_token_id exist
                tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token

            # add token and increase length by one
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
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
            decoded = input_ids.new(batch_size, sent_lengths.max().item()).fill_(pad_token_id)
        else:
            decoded = input_ids

        for hypo_idx, hypo in enumerate(input_ids):
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

    
    decoder = BertForMaskedLM(decoder_config)
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










