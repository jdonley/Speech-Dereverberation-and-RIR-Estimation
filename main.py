from argparse import ArgumentParser
from models.lightning_model import getModel
from datasets.reverb_speech_data import DareDataModule
import torch as t
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.profiler import AdvancedProfiler
from utils.utils import getConfig
from utils.progress_bar import getProgressBar
import random
import numpy as np
from WaveUnet.waveunet import Waveunet

random.seed(   getConfig()['random_seed'])
np.random.seed(getConfig()['random_seed'])
t.manual_seed( getConfig()['random_seed'])

def main(args):
    # ===========================================================
    # Configuration
    cfg = getConfig(config_path=args.config_path)
    
    # PyTorch Lightning Models
    model = getModel(**cfg['Model'])
    if model == None: # this happens when waveunet is chosen
        channels         = cfg['WaveUnet']['channels']
        kernel_size_down = cfg['WaveUnet']['kernel_size_down']
        kernel_size_up   = cfg['WaveUnet']['kernel_size_up']
        levels           = cfg['WaveUnet']['levels']
        feature_growth   = cfg['WaveUnet']['feature_growth']
        output_size      = cfg['WaveUnet']['output_size']
        sr               = cfg['WaveUnet']['sr']
        conv_type        = cfg['WaveUnet']['conv_type']
        res              = cfg['WaveUnet']['res']
        features         = cfg['WaveUnet']['features']
        instruments      = ["speech", "rir"]
        num_features     = [features*i for i in range(1, levels+1)] if feature_growth == "add" else \
                           [features*2**i for i in range(0, levels)]
        target_outputs   = int(output_size * sr)
        model            = Waveunet(channels, num_features, channels, instruments, kernel_size_down=kernel_size_down, kernel_size_up=kernel_size_up, target_output_size=target_outputs, conv_type=conv_type, res=res, separate=False)

    print("Using model " + model.name)
    # Data Module
    datamodule = DareDataModule(config=cfg)

    # Checkpoints
    ckpt_callback = ModelCheckpoint(
        **cfg['ModelCheckpoint'],
        filename = model.name + "-{epoch:02d}-{val_loss:.4f}",
        save_top_k=-1,
        every_n_epochs=1
    )
    
    # Learning Rate Monitor
    lr_monitor = LearningRateMonitor(**cfg['LearningRateMonitor'])

    # Strategy
    strategy = DDPStrategy(**cfg['DDPStrategy'])

    # Profiler
    profiler = AdvancedProfiler(**cfg['AdvancedProfiler'])
    
    # PyTorch Lightning Train
    trainer = pl.Trainer(
        **cfg['Trainer'],
        strategy=strategy,
        #profiler=profiler,
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