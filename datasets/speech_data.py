from torchaudio.datasets import LIBRISPEECH
from torch.utils.data import DataLoader
from pathlib import Path
from utils.utils import getConfig

def LibriSpeechDataset(config_path, type="train"):
    if type == "train":
        url = "train-clean-100"
    elif type == "val":
        url = "dev-clean"
    elif type == "test":
        url = "test-clean"
    else:
        url=""
    return LIBRISPEECH(
        Path(getConfig(config_path)['datasets_path']),
        url=url,
        download=True
        )

def LibriSpeechDataloader(config_path, type="train"):
    return DataLoader(LibriSpeechDataset(config_path, type=type))