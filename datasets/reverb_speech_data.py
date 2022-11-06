from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from utils import *
from utils import getConfig
from datasets.speech_data import LibriSpeechDataset
from datasets.rir_data import MitIrSurveyDataset
import librosa
from scipy import signal
import numpy as np

class DareDataset(Dataset):
    def __init__(self, type="train", split_train_val_test_p=[80,10,10], device='cuda'):

        self.type = type
        self.split_train_val_test_p = split_train_val_test_p
        self.device = device
        
        self.rir_dataset = MitIrSurveyDataset(type=self.type, device=device)
        self.speech_dataset = LibriSpeechDataset(type=self.type)

        self.eps = 10**-32

        # Approx. 4 seconds at 16kHz
        self.nfft = 511     # Makes it a nice 256 long stft (power of 2 for compute efficiency)
        self.nfrms = 256
        self.samplerate = self.speech_dataset[0][1]

        self.rir_duration = 2 * self.samplerate # the longest RIR is just under 2 seconds, so make them all that long

        self.nhop = 256
        self.num_speech_samples = 30 #0
        self.reverb_speech_duration = self.nfrms * (self.nfft+1)//2

    def __len__(self):
        return len(self.speech_dataset) * len(self.rir_dataset)

    def __getitem__(self, idx):
        idx_speech = idx % len(self.speech_dataset)
        idx_rir    = idx // len(self.speech_dataset)

        speech = self.speech_dataset[idx_speech][0].flatten()

        rir = self.rir_dataset[idx_rir].flatten()
        rir = rir[~np.isnan(rir)]

        rir = librosa.resample(rir,
            orig_sr=self.rir_dataset.samplerate,
            target_sr=self.samplerate,
            res_type='soxr_hq')

        reverb_speech = signal.convolve(speech, rir)

        reverb_speech = np.pad(
            reverb_speech,
            pad_width=(0, np.max((0,self.reverb_speech_duration - len(reverb_speech)))),
            )
        reverb_speech = reverb_speech[:self.reverb_speech_duration]
        
        speech = np.pad(
            speech,
            pad_width=(0, np.max((0,self.reverb_speech_duration - len(speech)))),
            )
        speech = speech[:self.reverb_speech_duration]

        reverb_speech_stft = librosa.stft(
            reverb_speech,
            n_fft=self.nfft,
            hop_length=self.nhop,
            win_length=self.nfft,
            window='hann'
            )

        np.seterr(divide = 'ignore')
        rs_mag = np.log(np.abs(reverb_speech_stft)) # Magnitude
        np.seterr(divide = 'warn')
        rs_mag[np.isinf(rs_mag)] = self.eps
        # Normalize to [-1,1]
        rs_mag = rs_mag - rs_mag.min()
        rs_mag = rs_mag / rs_mag.max() / 2 - 1

        reverb_speech = np.stack((rs_mag, np.angle(reverb_speech_stft)))

        speech_stft = librosa.stft(
            speech,
            n_fft=self.nfft,
            hop_length=self.nhop,
            win_length=self.nfft,
            window='hann'
            )

        np.seterr(divide = 'ignore')
        s_mag = np.log(np.abs(speech_stft)) # Magnitude
        np.seterr(divide = 'warn')
        s_mag[np.isinf(s_mag)] = self.eps
        # Normalize to [-1,1]
        s_mag = s_mag - s_mag.min()
        s_mag = s_mag / s_mag.max() / 2 - 1

        speech = np.stack((s_mag, np.angle(speech_stft)))
        
        rir = np.pad(
            rir,
            pad_width=(0, np.max((0,self.rir_duration - len(rir)))),
            )
            
        return reverb_speech, speech, rir

def DareDataloader(type="train"):
    return DataLoader(
        DareDataset(type),
        batch_size=getConfig()['batch_size'],
        shuffle=getConfig()['shuffle'] if type=="train" else False,
        drop_last=getConfig()['drop_last'],
        num_workers=getConfig()['num_workers'],
        pin_memory=getConfig()['pin_memory'],
        persistent_workers=getConfig()['persistent_workers'])

class DareDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
    def train_dataloader(self):
        return DareDataloader(type="train")
    def val_dataloader(self):
        return DareDataloader(type="val")
    def test_dataloader(self):
        return DareDataloader(type="test")