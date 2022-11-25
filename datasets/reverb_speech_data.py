from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from datasets.speech_data import LibriSpeechDataset
from datasets.rir_data import MitIrSurveyDataset
import librosa
from scipy import signal
import numpy as np

class DareDataset(Dataset):
    def __init__(self, config, type="train", split_train_val_test_p=[80,10,10], device='cuda'):

        self.model  = config['Model']['model_name']
        self.waveunet_input_samples  = 73721
        self.waveunet_output_samples = 32777
        self.type   = type
        self.split_train_val_test_p = split_train_val_test_p
        self.device = device
        
        self.rir_dataset = MitIrSurveyDataset(config, type=self.type, device=device)
        self.speech_dataset = LibriSpeechDataset(config, type=self.type)

        self.eps = 10**-32

        # Approx. 4 seconds at 16kHz
        self.nfft = 511     # Makes it a nice 256 long stft (power of 2 for compute efficiency)
        self.nfrms = 256
        self.samplerate = self.speech_dataset[0][1] # 16 kHz

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

        # Resample the rirs from 32 kHz to 16 kHz
        rir = librosa.resample(rir,
            orig_sr=self.rir_dataset.samplerate,
            target_sr=self.samplerate,
            res_type='soxr_hq')

        #print('**************************')
        #print('self.samplerate = ' + str(self.samplerate))
        #print('speech.shape = ' + str(speech.shape))
        #print('rir.shape = ' + str(rir.shape))


        reverb_speech = signal.convolve(speech, rir, method='fft')
        #print('speech.shape = ' + str(speech.shape))
        #print('reverb_speech.shape = ' + str(reverb_speech.shape))

        if self.model == 'Waveunet':
            reverb_speech = np.pad(
                reverb_speech,
                pad_width=(0, np.max((0,135141 - len(reverb_speech)))),
                )
            reverb_speech = reverb_speech[:135141] # expected input size given 15 up, 5 down filters and 2sec output

            speech = np.pad(
                speech,
                pad_width=(0, np.max((0,135141 - len(speech)))),
                )
            speech = speech[:135141]               # expected input size given 15 up, 5 down filters and 2sec output

            rir = np.pad(
                rir,
                pad_width=(0, np.max((0,32777 - len(rir)))),
                )            
            return reverb_speech, speech, rir 

        else:    
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
                
            return reverb_speech, speech, rir # 256 x 256 x 2 (mag, phase), 256 x 256 x 2, 32000

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
