import os
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

from argparse import ArgumentParser
from models.lightning_model import SpeechDAREUnet_v1, ErnstUnet, LitAutoEncoder
from datasets.reverb_speech_data import DareDataModule
import torch as t
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import getConfig
import random
import numpy as np

random.seed(   getConfig()['random_seed'])
np.random.seed(getConfig()['random_seed'])
t.manual_seed( getConfig()['random_seed'])

def main(args):
    # ===========================================================
    # Configuration
    cfg = getConfig(args.config_path)

    # PyTorch Lightning Models
    if args.model_name == "SpeechDAREUnet_v1":
        model = SpeechDAREUnet_v1(cfg['learning_rate'])
    elif args.model_name == "ErnstUnet":
        model = ErnstUnet(cfg['learning_rate'])

    # Data Module
    data_module = DareDataModule()

    # Checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor  = "val_loss",
        dirpath  = cfg['checkpoint_dirpath'],
        filename = model.name + "-{epoch:02d}-{val_loss:.2f}",
    )

    # PyTorch Lightning Train
    trainer = pl.Trainer(
        limit_train_batches    = cfg['train_batches'],
        limit_val_batches      = cfg['val_batches'],
        limit_test_batches     = cfg['test_batches'],
        max_epochs             = cfg['max_epochs'],
        log_every_n_steps      = cfg['log_every_n_steps'],
        accelerator            = cfg['accelerator'],
        devices                = cfg['devices'],
        strategy               = cfg['strategy'],
        callbacks              = [checkpoint_callback],
        profiler               = "simple"
        )

    trainer.fit(
        model      = model,
        datamodule = data_module
        )

    # ===========================================================
    # PyTorch Lightning Test
    trainer.test(
        model      = model,
        datamodule = data_module,
        ckpt_path  = "best"
        )
    
    return True

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="SpeechDAREUnet_v1", 
        help="SpeechDAREUnet_v1 or ErnstUnet")
    parser.add_argument("--config_path", type=str, default="config.yaml", 
        help="A full or relative path to a configuration yaml file. (default: config.yaml)")
        
    args = parser.parse_args()
    main(args)