from models.lightning_model import LitAutoEncoder
from datasets.reverb_speech_data import DareDataloader
import pytorch_lightning as pl
from utils import getConfig

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
        limit_train_batches = 1,
        limit_val_batches   = 1,
        limit_test_batches  = 1,
        max_epochs          = 1,
        accelerator         = 'cpu',
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
