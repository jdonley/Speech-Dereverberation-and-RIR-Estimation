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
    datamodule = DareDataModule(config_path=config_path)

    # Checkpoints
    ckpt_callback = ModelCheckpoint(
        **cfg['ModelCheckpoint'],
        filename = model.name + "-{epoch:02d}-{val_loss:.2f}",
    )

    # PyTorch Lightning Train
    trainer = pl.Trainer(**cfg['Trainer'], callbacks=[ckpt_callback])

    #trainer.fit(
    #    model=model,
    #    datamodule=datamodule
    #    )

    # ===========================================================
    # PyTorch Lightning Test
    #trainer.test(
    #    model=model,
    #    datamodule=datamodule
    #    ckpt_path="best"
    #    )
    
    return True


def test_answer():
    assert dummy_flow()
