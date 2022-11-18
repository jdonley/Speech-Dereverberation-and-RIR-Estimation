from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from datasets.speech_data import LibriSpeechDataset
from datasets.rir_data import MitIrSurveyDataset
import librosa
from scipy import signal
import numpy as np

class DareDataset(Dataset):
    def __init__(self, config, type="train", split_train_val_test_p=[80,10,10], device='cuda'):

        self.type = type
        self.split_train_val_test_p = split_train_val_test_p
        self.device = device
        
        self.rir_dataset = MitIrSurveyDataset(config, type=self.type, device=device)
        self.speech_dataset = LibriSpeechDataset(config, type=self.type)

        self.stft_format = config['stft_format']
        self.eps = 10**-32

        self.nfft = config['nfft']
        self.nfrms = config['nfrms']
        self.samplerate = self.speech_dataset[0][1]

        self.rir_duration = config['rir_duration']

        self.nhop = config['nhop']
        self.reverb_speech_duration = self.nfrms * self.nhop

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

        reverb_speech = signal.convolve(speech, rir, method='fft')

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
        
        # import matplotlib.pyplot as plt
        # plt.imshow(np.log(np.abs(reverb_speech_stft)), aspect='auto')
        # plt.show()
        # plt.close()


        if self.stft_format == 'magphase':
            np.seterr(divide = 'ignore')
            rs_mag = np.log(np.abs(reverb_speech_stft)) # Magnitude
            np.seterr(divide = 'warn')
            rs_mag[np.isinf(rs_mag)] = self.eps
            # Normalize to [-1,1]
            rs_mag = rs_mag - rs_mag.min()
            rs_mag = rs_mag / rs_mag.max() / 2 - 1

            reverb_speech = np.stack((rs_mag, np.angle(reverb_speech_stft)))
        
        elif self.stft_format == 'realimag':
            reverb_speech = np.stack((np.real(reverb_speech_stft), np.imag(reverb_speech_stft)))
            reverb_speech = reverb_speech - np.mean(reverb_speech)
            reverb_speech = reverb_speech / np.max(np.abs(reverb_speech))

        else:
            raise Exception("Unknown STFT format. Specify 'realimag' or 'magphase'.")

        speech_wav = speech / np.max(np.abs(speech)) - np.mean(speech)
        speech_stft = librosa.stft(
            speech,
            n_fft=self.nfft,
            hop_length=self.nhop,
            win_length=self.nfft,
            window='hann'
            )
        # import torch as t
        # s = t.tensor(np.real(speech_stft[None,:,:])).float() + 1j*t.tensor(np.imag(speech_stft[None,:,:])).float()
        # speech2 = np.squeeze(t.istft(s,self.nfft,hop_length=self.nhop,win_length=self.nfft,window=t.hann_window(self.nfft),length=speech_wav.shape[0]).numpy())
        # # print(speech-speech2)
        # import matplotlib.pyplot as plt
        # plt.plot(speech, label = "speech")
        # plt.plot(speech2, label = "speech2")
        # plt.plot(speech-speech2, label = "speech-speech2")
        # plt.show()
        # plt.close()

        if self.stft_format == 'magphase':
            np.seterr(divide = 'ignore')
            s_mag = np.log(np.abs(speech_stft)) # Magnitude
            np.seterr(divide = 'warn')
            s_mag[np.isinf(s_mag)] = self.eps
            # Normalize to [-1,1]
            s_mag = s_mag - s_mag.min()
            s_mag = s_mag / s_mag.max() / 2 - 1
            speech = np.stack((s_mag, np.angle(speech_stft)))

        elif self.stft_format == 'realimag':
            speech = np.stack((np.real(speech_stft), np.imag(speech_stft)))
            speech = speech - np.mean(speech)
            speech = speech / np.max(np.abs(speech))

        else:
            raise Exception("Unknown STFT format. Specify 'realimag' or 'magphase'.")
        
        rir = np.pad(
            rir,
            pad_width=(0, np.max((0,self.rir_duration - len(rir)))),
            )
        rir_fft = np.fft.rfft(rir)
        rir_fft = np.stack((np.real(rir_fft), np.imag(rir_fft)))
        rir_fft = rir_fft - np.mean(rir_fft)
        rir_fft = rir_fft / np.max(np.abs(rir_fft))
            
        return reverb_speech, speech, speech_wav, rir_fft[:,:,None]

def DareDataloader(config,type="train"):
    if type != "train":
        config['DataLoader']['shuffle'] = False
    return DataLoader(DareDataset(config,type),**config['DataLoader'])

class DareDataModule(LightningDataModule):
    def __init__(self,config):
        super().__init__()
        self.config = config
    def train_dataloader(self):
        return DareDataloader(type="train",config=self.config)
    def val_dataloader(self):
        return DareDataloader(type="val",config=self.config)
    def test_dataloader(self):
        return DareDataloader(type="test",config=self.config)
