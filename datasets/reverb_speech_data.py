import torch as t
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from utils import *
import numpy as np
from scipy import signal
from utils import getConfig
import soundfile as sf
from datasets.speech_data import LibriSpeechDataset
from datasets.rir_data import MitIrSurveyDataset

class DareDataset(Dataset):
    def __init__(self, type="train", split_train_val_test_p=[80,10,10], device='cuda'):
        
        self.type = type
        self.split_train_val_test_p = split_train_val_test_p
        self.device = device
        
        self.rir_dataset = MitIrSurveyDataset(type=self.type)
        self.speech_dataset = LibriSpeechDataset(type=self.type)



    def __len__(self):
        return len(self.speech_dataset) * len(self.rir_dataset)

    def __getitem__(self, idx):
        idx_speech = idx % len(self.speech_dataset)
        idx_rir    = idx // len(self.speech_dataset)

        speech = self.speech_dataset[idx_speech]
        rir = self.rir_dataset[idx_rir]

        reverb_speech = signal.fftconvolve(speech, rir, mode='full', axes=None)
        
        return reverb_speech

def DareDataloader(type="train"):
    return DataLoader(DareDataset(type))