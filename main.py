import os
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

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

def main():
    # ===========================================================
    # PyTorch Lightning Models
    #model = LitAutoEncoder(getConfig()['learning_rate'])
    #model = ErnstUnet(getConfig()['learning_rate'])
    model = SpeechDAREUnet_v1(getConfig()['learning_rate'])

    # Data Module
    data_module = DareDataModule()

    # Checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor  = "val_loss",
        dirpath  = getConfig()['checkpoint_dirpath'],
        filename = model.name + "-{epoch:02d}-{val_loss:.2f}",
    )

    # PyTorch Lightning Train
    trainer = pl.Trainer(
        limit_train_batches    = getConfig()['train_batches'],
        limit_val_batches      = getConfig()['val_batches'],
        limit_test_batches     = getConfig()['test_batches'],
        max_epochs             = getConfig()['max_epochs'],
        log_every_n_steps      = getConfig()['log_every_n_steps'],
        accelerator            = getConfig()['accelerator'],
        devices                = getConfig()['devices'],
        strategy               = getConfig()['strategy'],
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
    main()