import torchaudio
from torch import utils
from pathlib import Path
from utils import *

def LibriSpeechDataset(type="train"):
    if type == "train":
        url = "train-clean-100"
    elif type == "val":
        url = "dev-clean"
    elif type == "test":
        url = "test-clean"
    return torchaudio.datasets.LIBRISPEECH(
        Path(getConfig()['datasets_path']),
        url=url,
        download=True
        )

def LibriSpeechDataloader(type="train"):
    return utils.data.DataLoader(LibriSpeechDataset(type))