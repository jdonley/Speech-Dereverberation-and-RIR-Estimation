import torch as t
import torchaudio as ta
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
        
        self.rir_dataset = MitIrSurveyDataset(type=self.type, device=device)
        self.speech_dataset = LibriSpeechDataset(type=self.type)

        # Approx. 4 seconds at 16kHz
        self.nfft = 511     # Makes it a nice 256 long stft (power of 2 for compute efficiency)
        self.nfrms = 256
        self.samplerate = self.speech_dataset[0][1]

        self.rir_duration = 2 * self.samplerate # the longest RIR is just under 2 seconds, so make them all that long

        self.nhop = 256
        self.num_speech_samples = 30 #0
        self.reverb_speech_duration = self.nfrms * (self.nfft+1)//2

        self.resampler = ta.transforms.Resample(
            orig_freq=self.rir_dataset.samplerate,
            new_freq=self.samplerate)
        self.resampler.kernel = self.resampler.kernel.to(device)

        #self.reverb_speech = np.empty((
        #    self.num_speech_samples * len(self.rir_dataset),
        #    (self.nfft+1)//2,
        #    self.nfrms,
        #    2))
        #self.reverb_speech[:] = np.nan
        #self.reverb_speech = t.tensor(self.reverb_speech, dtype=t.float).to(self.device)

        #self.speech = np.empty((
        #    self.num_speech_samples * len(self.rir_dataset),
        #    (self.nfft+1)//2,
        #    self.nfrms,
        #    2))
        #self.speech[:] = np.nan
        #self.speech = t.tensor(self.speech, dtype=t.float).to(self.device)

        #self.rir = np.empty((
        #    30 * len(self.rir_dataset), 
        #    self.rir_duration))
        #self.rir[:] = np.nan
        #self.rir = t.tensor(self.rir, dtype=t.float).to(self.device)

    def __len__(self):
        return len(self.speech_dataset) * len(self.rir_dataset)

    def __getitem__(self, idx):
        #if t.all(t.isnan(self.reverb_speech[idx,:])):
        idx_speech = idx % len(self.speech_dataset)
        idx_rir    = idx // len(self.speech_dataset)

        speech = self.speech_dataset[idx_speech][0].to(self.device).flatten()


        rir = self.rir_dataset[idx_rir].flatten()
        rir = rir[~rir.isnan()]
        rir = self.resampler(rir) # downsample the RIRs to match the speech samplerate
        
        reverb_speech = t.nn.functional.conv1d(
            speech.view(1,1,-1),
            t.flip(rir,(0,)).view(1,1,-1),
            padding=len(rir) - 1
            ).view(-1)

        reverb_speech = t.nn.functional.pad(
            reverb_speech,
            pad=(0, self.reverb_speech_duration - len(reverb_speech)),
            mode="constant", value=0
            )
        
        reverb_speech = reverb_speech[:self.reverb_speech_duration]

        speech = t.nn.functional.pad(
            speech.to(self.device),
            pad=(0, self.reverb_speech_duration - len(speech)),
            mode="constant", value=0
            )
        
        speech = speech[:self.reverb_speech_duration]

        reverb_speech_stft = t.stft(
            reverb_speech,
            n_fft=self.nfft,
            hop_length=self.nhop,
            window=t.hann_window(self.nfft).to(self.device),
            normalized=True,
            return_complex=True
            )
        #self.reverb_speech[idx,:,:,0] = reverb_speech_stft.abs()
        #self.reverb_speech[idx,:,:,1] = reverb_speech_stft.angle()
        #reverb_speech = t.stack((reverb_speech_stft.abs(), reverb_speech_stft.angle()))
        reverb_speech = reverb_speech_stft.abs()
        speech_stft = t.stft(
            speech,
            n_fft=self.nfft,
            hop_length=self.nhop,
            window=t.hann_window(self.nfft).to(self.device),
            normalized=True,
            return_complex=True
            )
        #self.speech[idx,:,:,0] = speech_stft.abs()
        #self.speech[idx,:,:,1] = speech_stft.angle()
        speech = t.stack((speech_stft.abs(), speech_stft.angle()))
        
        rir = t.nn.functional.pad( # pad the RIR to 2s if shorter
            rir,
            pad=(0, self.rir_duration - len(rir)),
            mode="constant", value=0
        )
        
        #return self.reverb_speech[idx,:,:,:], self.speech[idx,:,:,:]
        return reverb_speech, speech, rir

def DareDataloader(type="train"):
    return DataLoader(
        DareDataset(type),
        batch_size =getConfig()['batch_size'],
        num_workers=getConfig()['num_workers'],
        persistent_workers=getConfig()['persistent_workers'])