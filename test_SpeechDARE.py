from models.lightning_model import *
#from datasets.reverb_speech_data import DareDataModule
#import pytorch_lightning as pl
#from utils import getTestConfig

def dummy_flow():
    # ===========================================================
    # PyTorch Lightning Models
    model = LitAutoEncoder()
    model = ErnstUnet()
    model = SpeechDAREUnet_v1()

    # Data Module
    #data_module = DareDataModule()

    # PyTorch Lightning Train
    #trainer = pl.Trainer(
    #    limit_train_batches = getTestConfig()['train_batches'],
    #    limit_val_batches   = getTestConfig()['val_batches'],
    #    limit_test_batches  = getTestConfig()['test_batches'],
    #    max_epochs          = getTestConfig()['max_epochs'],
    #    log_every_n_steps   = getTestConfig()['log_every_n_steps'],
    #    accelerator         = getTestConfig()['accelerator'],
    #    devices             = getTestConfig()['devices'],
    #    strategy            = getTestConfig()['strategy']
    #    )

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
