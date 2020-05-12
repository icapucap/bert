from time import time
from pathlib import Path
import torch
import numpy as np
from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.callbacks import ModelCheckpoint
import os

from pytorch_lightning.loggers import TensorBoardLogger
logger = TensorBoardLogger("/content/itr/lightning_logs", name="my_model")



# from data import IndicDataset, PadSequence
import model as M


def gen_model_loaders(config):
    model, tokenizers = M.build_model(config)
    return model, tokenizers

def init_seed():
    seed_val = 42
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


# from train_util import run_train
from config import replace, preEnc, preEncDec
from model import build_model
def main():
    init_seed()
    rconf = preEncDec
    model= build_model(rconf)

    checkpoint_callback = ModelCheckpoint(
    filepath=os.getcwd(),  #saves checkpoint in the root dir
    save_top_k=1,   #saves the best model
    verbose=True,
    monitor='val_loss',
    mode='min'
    )

    # writer = SummaryWriter(rconf.log_dir)
    # train_losses, val_losses, val_accs = run_train(rconf, model, train_loader, eval_loader, writer)
    #
    trainer = Trainer(max_epochs=3,logger= logger,log_save_interval=1,checkpoint_callback=checkpoint_callback)    

    #trainer.save_checkpoint('./my_checkpoint.ckpt') #for manually saving checkpoint
    #trainer = Trainer(resume_from_checkpoint=PATH,max_epochs=3) #path of the checkpoint file from where you want to continue training
    
    trainer.fit(model)
    # model.save(tokenizers, rconf.model_output_dirs)

if __name__ == '__main__':
    #preproc_data()
    main()


#tensorboard --logdir /content/itr/lightning_logs
#links to tensorboard





