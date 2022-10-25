from genericpath import isdir
import torch as t
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from utils import *
import numpy as np
from utils import getConfig
import soundfile as sf
import glob
import os
import requests
import zipfile
from tqdm import tqdm
import shutil

class MitIrSurveyDataset(Dataset):
    def __init__(self, type="train", split_train_val_test_p=[80,10,10], device='cuda', download=True):
        self.root_dir = Path(getConfig()['datasets_path'],'MIT_IR_Survey')
        
        if download and not os.path.isdir(str(self.root_dir)): # If the path doesn't exist, download the dataset if set to true
            self.download_mit_ir_survey(self.root_dir)

        self.max_data_len = 270 # This is supposed to be 271 but there is an IR missing in the dataset

        self.samplerate = 32000

        self.type = type
        self.split_train_val_test_p = split_train_val_test_p
        self.split_train_val_test = np.int32(np.round( np.array(self.split_train_val_test_p)/100 * self.max_data_len ))
        self.split_edge = np.cumsum(np.hstack((0,self.split_train_val_test_p)))
        np.random.seed(getConfig()['random_seed'])
        self.idx_rand = np.random.permutation(range(self.max_data_len))

        if self.type == "train":
            self.split = self.idx_rand[self.split_edge[0]:self.split_edge[1]]
        elif self.type == "val":
            self.split = self.idx_rand[self.split_edge[1]:self.split_edge[2]]
        elif self.type == "test":
            self.split = self.idx_rand[self.split_edge[2]:self.split_edge[3]]

        self.split_filenames = [glob.glob(str(Path(self.root_dir,"*")))[i] for i in self.split]

        self.device = device
        
        max_rir_len = 63971 # max num samples
        self.irs = np.empty((len(self.split), max_rir_len)) # VGGish features are 10 x 128
        self.irs[:] = np.nan
        self.irs = t.tensor(self.irs, dtype=t.float).to(self.device)

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        if t.all(t.isnan(self.irs[idx,:])):
            filename = self.split_filenames[idx]
            #audio_path = os.path.join(self.audio_dir, filename[0:3], filename + ".ogg")
            audio_data, samplerate = sf.read(filename)
            if samplerate != self.samplerate:
                raise Exception("The samplerate of the audio in the dataset is not 32kHz.")
            
            self.irs[idx,:len(audio_data)] = t.tensor(audio_data, dtype=t.float).to(self.device)
        
        return self.irs[idx,:]

    def download_mit_ir_survey(self, local_path):
        url = "https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip"

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(str(local_path)+".zip", 'wb') as f:
                for chunk in tqdm(r.iter_content(chunk_size=8192), total=1424, desc="Downloading MIT IR Survey"): 
                    f.write(chunk)

        with zipfile.ZipFile(str(local_path)+".zip", 'r') as zip_ref:
            zip_ref.extractall(str(local_path))

        files_list = os.listdir(str(Path(local_path,"Audio")))
        for file in files_list:
            if file != ".DS_Store":
                os.rename(str(Path(local_path,"Audio",file)), str(Path(local_path,file)))
        
        for dir in ([d[0] for d in os.walk(str(local_path))])[1:]:
            if os.path.isdir(dir):
                shutil.rmtree(dir)
        
        print("Download complete.")

        return True

def MitIrSurveyDataloader(type="train"):
    return DataLoader(MitIrSurveyDataset(type))