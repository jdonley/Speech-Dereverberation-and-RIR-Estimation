import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.lightning_model import *
from datasets.reverb_speech_data import DareDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
import pytorch_lightning as pl
from utils.utils import getTestConfig
from utils.progress_bar import getProgressBar

def dummy_flow():
    cfg = getTestConfig()
    # ===========================================================
    # PyTorch Lightning Models
    model = LitAutoEncoder()
    model = ErnstUnet()
    model = SpeechDAREUnet_v1()

    # Data Module
    datamodule = DareDataModule(config=cfg)

    # Checkpoints
    ckpt_callback = ModelCheckpoint(
        **cfg['ModelCheckpoint'],
        filename = model.name + "-{epoch:02d}-{val_loss:.2f}",
    )

    # Strategy
    strategy = DDPStrategy(**cfg['DDPStrategy'])

    # PyTorch Lightning Train
    trainer = pl.Trainer(
        **cfg['Trainer'],
        strategy=strategy,
        callbacks=[ckpt_callback,getProgressBar(cfg)]
        )

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
