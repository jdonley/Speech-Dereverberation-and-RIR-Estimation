from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import soundfile as sf
import glob
import os
import requests
import zipfile
from tqdm import tqdm
import shutil
#import pdb

class MitIrSurveyDataset(Dataset):
    def __init__(self, config, type="train", split_train_val_test_p=[80,10,10], device='cuda', download=True):
        self.root_dir = Path(os.path.expanduser(config['datasets_path']),'MIT_IR_Survey')
        self.type = type

        if download and not os.path.isdir(str(self.root_dir)): # If the path doesn't exist, download the dataset if set to true
            self.download_mit_ir_survey(self.root_dir)

        self.max_data_len = 270 # This is supposed to be 271 but there is an IR missing in the dataset
        self.samplerate = 32000

        self.split_train_val_test_p = np.array(np.int16(split_train_val_test_p))
        self.split_train_val_test = np.int16(np.round( np.array(self.split_train_val_test_p)/100 * self.max_data_len ))
        #self.split_edge = np.cumsum(np.concatenate(([0],self.split_train_val_test_p)), axis=0)
        self.split_edge = np.cumsum(np.concatenate(([0],self.split_train_val_test)), axis=0)
        self.idx_rand = np.random.RandomState(seed=config['random_seed']).permutation(self.max_data_len)
        print("self.split_train_val_test = " + str(self.split_train_val_test))

        split = []
        #pdb.set_trace()
        if self.type == "train":
            split = self.idx_rand[self.split_edge[0]:self.split_edge[1]]
        elif self.type == "val":
            split = self.idx_rand[self.split_edge[1]:self.split_edge[2]]
        elif self.type == "test":
            split = self.idx_rand[self.split_edge[2]:self.split_edge[3]]
            print("edges: " + str(self.split_edge[2]) + ":" + str(self.split_edge[3]))
            print("split = " + str(split))

        files = glob.glob(str(Path(self.root_dir,"*")))
        self.split_filenames = [files[i] for i in split]
        self.device = device

    def __len__(self):
        return len(self.split_filenames)

    def __getitem__(self, idx):
        filename = self.split_filenames[idx]
        audio_data, samplerate = sf.read(filename)
        if samplerate != self.samplerate:
            raise Exception("The samplerate of the audio in the dataset is not 32kHz.")
        
        return audio_data

    def download_mit_ir_survey(self, local_path):
        url = "https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip"
        local_path = os.path.expanduser(local_path)

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

def MitIrSurveyDataloader(config_path, type="train"):
    return DataLoader(MitIrSurveyDataset(config_path, type=type))