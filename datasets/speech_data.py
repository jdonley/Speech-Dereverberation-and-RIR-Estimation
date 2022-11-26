from torchaudio.datasets import LIBRISPEECH
from torch.utils.data import DataLoader
from pathlib import Path
import os

def LibriSpeechDataset(config, type="train"):
    if type == "train":
        url = "train-clean-100"
    elif type == "val":
        url = "dev-clean"
    elif type == "test":
        url = "test-clean"
    else:
        url=""
    return LIBRISPEECH(
        Path(os.path.expanduser(config['datasets_path'])),
        url=url,
        download=True
        )

def LibriSpeechDataloader(config, type="train"):
    return DataLoader(LibriSpeechDataset(config, type=type))