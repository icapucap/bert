from time import time
from pathlib import Path
import torch
import numpy as np
from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter







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
    # writer = SummaryWriter(rconf.log_dir)
    # train_losses, val_losses, val_accs = run_train(rconf, model, train_loader, eval_loader, writer)
    trainer = Trainer()
    trainer.fit(model)
    # model.save(tokenizers, rconf.model_output_dirs)

if __name__ == '__main__':
    #preproc_data()
    main()








