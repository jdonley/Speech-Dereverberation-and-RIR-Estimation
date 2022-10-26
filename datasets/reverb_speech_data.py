import torch as t
from torch.utils.data import Dataset, DataLoader
from utils import *
from scipy import signal
from utils import getConfig
from datasets.speech_data import LibriSpeechDataset
from datasets.rir_data import MitIrSurveyDataset
import librosa
import numpy as np

class DareDataset(Dataset):
    def __init__(self, type="train", split_train_val_test_p=[80,10,10], device='cuda'):
        
        self.type = type
        self.split_train_val_test_p = split_train_val_test_p
        self.device = device
        
        self.rir_dataset = MitIrSurveyDataset(type=self.type, device="cpu")
        self.speech_dataset = LibriSpeechDataset(type=self.type)

        self.samplerate = 16000
        self.reverb_speech_duration = 10 * self.samplerate

        self.reverb_speech = np.empty((
            30 * len(self.rir_dataset), 
            self.reverb_speech_duration))
        self.reverb_speech[:] = np.nan
        self.reverb_speech = t.tensor(self.reverb_speech, dtype=t.float).to(self.device)

        self.speech = np.empty((
            30 * len(self.rir_dataset), 
            self.reverb_speech_duration))
        self.speech[:] = np.nan
        self.speech = t.tensor(self.speech, dtype=t.float).to(self.device)

    def __len__(self):
        return len(self.speech_dataset) * len(self.rir_dataset)

    def __getitem__(self, idx):
        if t.all(t.isnan(self.reverb_speech[idx,:])):
            idx_speech = idx % len(self.speech_dataset)
            idx_rir    = idx // len(self.speech_dataset)

            speech,fs_speech = self.speech_dataset[idx_speech][0:2]
            speech = speech.flatten()
            rir = self.rir_dataset[idx_rir].flatten()
            rir = rir[~rir.isnan()]
            fs_rir = self.rir_dataset.samplerate

            rir = librosa.resample(rir.numpy(), orig_sr=fs_rir, target_sr=fs_speech)
            rir = t.tensor(rir, dtype=t.float).to(self.rir_dataset.device)

            reverb_speech = signal.fftconvolve(speech, rir, mode='full', axes=None)
            reverb_speech = t.tensor(reverb_speech, dtype=t.float).to(self.device)

            reverb_speech = t.nn.functional.pad(
                reverb_speech,
                pad=(0, self.reverb_speech_duration - len(reverb_speech)),
                mode="constant", value=0
                )
            
            reverb_speech = reverb_speech[:self.reverb_speech_duration]

            speech = t.nn.functional.pad(
                speech,
                pad=(0, self.reverb_speech_duration - len(speech)),
                mode="constant", value=0
                )
            
            speech = speech[:self.reverb_speech_duration]

            self.reverb_speech[idx,:] = reverb_speech
            self.speech[idx,:] = speech
        
        return self.reverb_speech[idx,:], self.speech[idx,:]

def DareDataloader(type="train", batch_size=128):
    return DataLoader(DareDataset(type), batch_size=batch_size)