import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.lightning_model import *
from datasets.reverb_speech_data import DareDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from utils.utils import getTestConfig

def dummy_flow():
    cfg, config_path = getTestConfig()
    # ===========================================================
    # PyTorch Lightning Models
    model = LitAutoEncoder()
    model = ErnstUnet()
    model = SpeechDAREUnet_v1()

    # Data Module
    data_module = DareDataModule(config_path)

    # Checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=cfg['checkpoint_dirpath'],
        filename=model.name+"-{epoch:02d}-{val_loss:.2f}",
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
        callbacks=[checkpoint_callback]
        )

    #trainer.fit(
    #    model=model,
    #    datamodule=data_module
    #    )

    # ===========================================================
    # PyTorch Lightning Test
    #trainer.test(
    #    model=model,
    #    datamodule=data_module
    #    ckpt_path="best"
    #    )
    
    return True


def test_answer():
    assert dummy_flow()
