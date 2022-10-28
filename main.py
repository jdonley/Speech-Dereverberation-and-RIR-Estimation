from models.lightning_model import LitAutoEncoder
from datasets.reverb_speech_data import DareDataloader
import pytorch_lightning as pl
from utils import getConfig

def main():
    # ===========================================================
    # PyTorch Lightning Models
    autoencoder = LitAutoEncoder()

    # Data Loaders
    train_loader = DareDataloader("train")
    val_loader   = DareDataloader("val")
    test_loader  = DareDataloader("test")

    # PyTorch Lightning Train
    trainer = pl.Trainer(
        limit_train_batches = getConfig()['train_batches'],
        limit_val_batches   = getConfig()['val_batches'],
        limit_test_batches  = getConfig()['test_batches'],
        max_epochs          = getConfig()['max_epochs'],
        log_every_n_steps   = getConfig()['log_every_n_steps'],
        accelerator         = getConfig()['accelerator'],
        devices             = getConfig()['devices'],
        strategy            = getConfig()['strategy'],
        profiler="simple"
        )

    trainer.fit(
        model=autoencoder,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
        )

    # ===========================================================
    # PyTorch Lightning Test
    trainer.test(
        model=autoencoder,
        dataloaders=test_loader,
        ckpt_path="best"
        )
    
    return True

if __name__ == "__main__":
        main()