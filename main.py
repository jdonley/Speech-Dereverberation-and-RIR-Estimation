from argparse import ArgumentParser
from models.lightning_model import getModel
from datasets.reverb_speech_data import DareDataModule
import torch as t
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies.ddp import DDPStrategy
from utils.utils import getConfig
from utils.progress_bar import getProgressBar
import random
import numpy as np

random.seed(   getConfig()['random_seed'])
np.random.seed(getConfig()['random_seed'])
t.manual_seed( getConfig()['random_seed'])

def main(args):
    # ===========================================================
    # Configuration
    cfg = getConfig(config_path=args.config_path)

    # PyTorch Lightning Models
    model = getModel(**cfg['Model'])

    # Data Module
    datamodule = DareDataModule(config=cfg)

    # Checkpoints
    ckpt_callback = ModelCheckpoint(
        **cfg['ModelCheckpoint'],
        filename = model.name + "-{epoch:02d}-{val_loss:.2f}",
    )
    
    # Learning Rate Monitor
    lr_monitor = LearningRateMonitor(**cfg['LearningRateMonitor'])

    # Strategy
    strategy = DDPStrategy(**cfg['DDPStrategy'])

    # PyTorch Lightning Train
    trainer = pl.Trainer(
        **cfg['Trainer'],
        strategy=strategy,
        callbacks=[ckpt_callback,lr_monitor,getProgressBar(cfg)]
        )

    trainer.fit(
        model      = model,
        datamodule = datamodule
        )

    # ===========================================================
    # PyTorch Lightning Test
    trainer.test(
        model      = model,
        datamodule = datamodule,
        ckpt_path  = "best"
        )
    
    return True

if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("--config_path", type=str, default="config.yaml", 
        help="A full or relative path to a configuration yaml file. (default: config.yaml)")
        
    args = parser.parse_args()
    main(args)