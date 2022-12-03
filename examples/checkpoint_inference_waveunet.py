import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
#from models.lightning_model import *
from WaveUnet.waveunet import *
from WaveUnet.crop import centre_crop
from datasets.reverb_speech_data import DareDataset
import numpy as np
from utils.utils import getConfig
import matplotlib.pyplot as plt
import argparse
import torch as t
import torchaudio as ta
import platform

p = argparse.ArgumentParser()
p.add_argument('ckpt_path')
p.add_argument('config_path')

def run_checkpoint(config_path,ckpt_path):
    cfg = getConfig(config_path)
    channels         = cfg['WaveUnet']['channels']
    kernel_size_down = cfg['WaveUnet']['kernel_size_down']
    kernel_size_up   = cfg['WaveUnet']['kernel_size_up']
    levels           = cfg['WaveUnet']['levels']
    feature_growth   = cfg['WaveUnet']['feature_growth']
    output_size      = cfg['WaveUnet']['output_size']
    sr               = cfg['WaveUnet']['sr']
    conv_type        = cfg['WaveUnet']['conv_type']
    res              = cfg['WaveUnet']['res']
    features         = cfg['WaveUnet']['features']
    instruments      = ["speech", "rir"]
    num_features     = [features*i for i in range(1, levels+1)] if feature_growth == "add" else \
                       [features*2**i for i in range(0, levels)]
    target_outputs   = int(output_size * sr)

    model = Waveunet.load_from_checkpoint(ckpt_path, None, None, True, num_inputs=channels, num_channels=num_features, num_outputs=channels, instruments=instruments, kernel_size_down=kernel_size_down, kernel_size_up=kernel_size_up, target_output_size=target_outputs, conv_type=conv_type, res=res)
    model.eval()

    test_loader  = DareDataset(cfg,"test") # was test (as it should be)
    example = test_loader[0] # was 0
    x = example[0]
    y = example[1]
    z = example[2]

    #print("x.shape = " + str(x.shape))
    
    x = t.tensor(x[None, None, :],dtype=t.float)
    y = t.tensor(y[None, None, :],dtype=t.float)
     
    prediction = model.forward(x)

    #loss   = nn.functional.mse_loss(prediction[0,0,:,:].squeeze().detach().numpy(), y[0,:,:].squeeze())
    #loss = np.mean(np.square(np.subtract(prediction[0,0,:,:].squeeze().detach().numpy(), y[0,:,:].squeeze())))
    #print("************************")
    #print("prediction.shape = " + str(prediction.shape))
    #print("clean.shape = " + str(y.shape))
    
    #print("loss = " + str(loss))
    #print("************************")

    x = centre_crop(x, prediction["speech"])
    y = centre_crop(y, prediction["speech"])

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
    fig.set_size_inches(24, 4.8)
    #fig.tight_layout()
    ax1.plot(x.squeeze().detach().numpy())
    ax2.plot(y.squeeze().detach().numpy())
    ax3.plot(z)
    ax4.plot(prediction["speech"].squeeze().detach().numpy())
    ax5.plot(prediction["rir"].squeeze().detach().numpy())
    ax1.title.set_text("Cropped Reverb Speech")
    ax2.title.set_text("Cropped Clean Speech")
    ax3.title.set_text("GT RIR")
    ax4.title.set_text("Predicted Clean Speech")
    ax5.title.set_text("Predicted RIR")
    ax3.set_xlim(2000, 6000)

    plt.show()

    if platform.system() == 'Windows':
        revSpeech   = 'C:\\Users\\PCALAMIA\\Dropbox (Meta)\\StanfordAI\\NN\\Project\\revSpeech_11_lr1e-4.wav'
        cleanSpeech = 'C:\\Users\\PCALAMIA\\Dropbox (Meta)\\StanfordAI\\NN\\Project\\cleanSpeech_11_lr1e-4.wav'
        predSpeech  = 'C:\\Users\\PCALAMIA\\Dropbox (Meta)\\StanfordAI\\NN\\Project\\predictedSpeech_11_lr1e-4.wav'
    else:
        #revSpeech   = '/home/pcalamia/Dropbox (Meta)/StanfordAI/NN/Project/revSpeech_0_lr1e-4.wav'
        #cleanSpeech = '/home/pcalamia/Dropbox (Meta)/StanfordAI/NN/Project/cleanSpeech_0_lr1e-4.wav'
        #predSpeech  = '/home/pcalamia/Dropbox (Meta)/StanfordAI/NN/Project/predictedSpeech_0_lr1e-4.wav'
        revSpeech   = '/home/pcalamia/temp/revSpeech.wav'
        cleanSpeech = '/home/pcalamia/temp/cleanSpeech.wav'
        predSpeech  = '/home/pcalamia/temp/predictedSpeech.wav'

    outWav      = prediction["speech"].squeeze().detach().numpy()[None, :]
    
    #print("outWav.shape = " + str(outWav.shape))
    #print(outWav.dtype)
    ta.save(predSpeech, t.tensor(0.99*outWav/(np.max(np.abs(outWav)))), 16000)
    ta.save(revSpeech, x.squeeze()[None, :], 16000)
    ta.save(cleanSpeech, y.squeeze()[None, :], 16000)

if __name__=='__main__':
    args = p.parse_args()
    run_checkpoint(**vars(args))