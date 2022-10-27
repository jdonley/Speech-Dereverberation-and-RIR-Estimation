from models.lightning_model import LitAutoEncoder
from datasets.reverb_speech_data import DareDataloader
import pytorch_lightning as pl
from utils import getTestConfig

def dummy_flow():
    # ===========================================================
    # PyTorch Lightning Models
    autoencoder = LitAutoEncoder()

    # Data Loaders
    train_loader = DareDataloader("train")
    val_loader   = DareDataloader("val")
    test_loader  = DareDataloader("test")

    # PyTorch Lightning Train
    trainer = pl.Trainer(
        limit_train_batches = getTestConfig()['train_batches'],
        limit_val_batches   = getTestConfig()['val_batches'],
        limit_test_batches  = getTestConfig()['test_batches'],
        max_epochs          = getTestConfig()['max_epochs'],
        log_every_n_steps   = getTestConfig()['log_every_n_steps'],
        accelerator         = getTestConfig()['accelerator'],
        devices             = getTestConfig()['devices'],
        strategy            = getTestConfig()['strategy']
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


def test_answer():
    assert dummy_flow()
