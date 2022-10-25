from torch.utils.data import Dataset, DataLoader
from utils import *
from scipy import signal
from utils import getConfig
from datasets.speech_data import LibriSpeechDataset
from datasets.rir_data import MitIrSurveyDataset
import librosa

class DareDataset(Dataset):
    def __init__(self, type="train", split_train_val_test_p=[80,10,10], device='cuda'):
        
        self.type = type
        self.split_train_val_test_p = split_train_val_test_p
        self.device = device
        
        self.rir_dataset = MitIrSurveyDataset(type=self.type, device="cpu")
        self.speech_dataset = LibriSpeechDataset(type=self.type)



    def __len__(self):
        return len(self.speech_dataset) * len(self.rir_dataset)

    def __getitem__(self, idx):
        idx_speech = idx % len(self.speech_dataset)
        idx_rir    = idx // len(self.speech_dataset)

        speech,fs_speech = self.speech_dataset[idx_speech][0:2]
        speech = speech.flatten()
        rir = self.rir_dataset[idx_rir].flatten()
        rir = rir[~rir.isnan()]
        fs_rir = self.rir_dataset.samplerate

        rir = librosa.resample(rir, orig_sr=fs_rir, target_sr=fs_speech)

        reverb_speech = signal.fftconvolve(speech, rir, mode='full', axes=None)
        
        return reverb_speech

def DareDataloader(type="train"):
    return DataLoader(DareDataset(type))