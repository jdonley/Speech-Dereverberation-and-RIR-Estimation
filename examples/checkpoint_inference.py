import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from models.lightning_model import *
from datasets.reverb_speech_data import DareDataset
import matplotlib.pyplot as plt
import argparse
p = argparse.ArgumentParser()
p.add_argument('ckpt_path')

def run_checkpoint(ckpt_path):
    model = SpeechDAREUnet_v1.load_from_checkpoint(ckpt_path)
    model.eval()

    test_loader  = DareDataset("test")
    example = test_loader[0]
    x = example[0]
    y = example[1]
    print(x)
    prediction = model.predict(t.tensor(x[:,:,:,None],dtype=t.float).permute((0,3,1,2)))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(x[0,:,:].squeeze())
    ax2.imshow(y[0,:,:].squeeze())
    ax3.imshow(prediction[0,0,:,:].squeeze().detach().numpy())
    plt.show()

if __name__=='__main__':
    args = p.parse_args()
    run_checkpoint(**vars(args))