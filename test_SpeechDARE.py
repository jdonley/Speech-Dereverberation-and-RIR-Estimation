from models.lightning_model import *
from datasets.reverb_speech_data import DareDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from utils import getConfig

def dummy_flow():
    # ===========================================================
    # PyTorch Lightning Models
    model = LitAutoEncoder()
    model = ErnstUnet()
    model = SpeechDAREUnet_v1()

    # Data Module
    data_module = DareDataModule()

    # Checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=getConfig()['checkpoint_dirpath'],
        filename=model.name+"-{epoch:02d}-{val_loss:.2f}",
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
