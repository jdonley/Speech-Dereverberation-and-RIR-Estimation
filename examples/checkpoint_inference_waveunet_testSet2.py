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
    
    nExamples = test_loader.__len__()
    #nExamples = 100
    rcMSE     = np.zeros(nExamples)
    rpMSE     = np.zeros(nExamples)
    cpMSE     = np.zeros(nExamples)
    errAbs    = np.zeros(nExamples)    
    errPhase  = np.zeros(nExamples)    

    for ii in range(nExamples): 
    
        if ii % 10 == 0:
            print(ii)
        
        example = test_loader[ii] # was 0
        x = example[0] # reverberant speech
        y = example[1] # clean speech
        z = example[2] # time-domain rir

        # Create freq-domain rir for Jacob's losses
        # Ground truth rir fft
        gt_rir_fft = np.fft.rfft(z)
        gt_rir_fft = np.stack((np.real(gt_rir_fft), np.imag(gt_rir_fft)))
        gt_rir_fft = gt_rir_fft - np.mean(gt_rir_fft)
        gt_rir_fft = gt_rir_fft / np.max(np.abs(gt_rir_fft))
        gt_rir_fft = gt_rir_fft[None, :, :] # add the batch dimension
        gt_rir_fft = gt_rir_fft[:, :, :, None] # add a time dimension (?)

        x = t.tensor(x[None, None, :],dtype=t.float)
        y = t.tensor(y[None, None, :],dtype=t.float)
     
        prediction = model.forward(x)

        pred_rir_fft = np.fft.rfft(prediction["rir"].squeeze().detach().numpy())
        pred_rir_fft = np.stack((np.real(pred_rir_fft), np.imag(pred_rir_fft)))
        pred_rir_fft = pred_rir_fft - np.mean(pred_rir_fft)
        pred_rir_fft = pred_rir_fft / np.max(np.abs(pred_rir_fft))
        pred_rir_fft = pred_rir_fft[None, :, :] # add the batch dimension
        pred_rir_fft = pred_rir_fft[:, :, :, None] # add a time dimension (?)

        rcMSE[ii] = nn.functional.mse_loss(centre_crop(x, prediction["speech"]), centre_crop(y, prediction["speech"]))
        rpMSE[ii] = nn.functional.mse_loss(centre_crop(x, prediction["speech"]), prediction["speech"])
        cpMSE[ii] = nn.functional.mse_loss(centre_crop(y, prediction["speech"]), prediction["speech"])

        z = t.tensor(z[None,  :],dtype=t.float)
        gt_rir_fft = t.tensor(gt_rir_fft,dtype=t.float)
        pred_rir_fft = t.tensor(pred_rir_fft,dtype=t.float)

        errAbs[ii], errPhase[ii] = \
            compute_losses(gt_rir_fft, z, pred_rir_fft, "test")

    # MSE histograms
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_size_inches(12, 4.8)
    ax1.hist(rcMSE, 100, density=False, facecolor='g', alpha=0.75)
    ax2.hist(rpMSE, 100, density=False, facecolor='g', alpha=0.75)
    ax3.hist(cpMSE, 100, density=False, facecolor='g', alpha=0.75)
    ax1.title.set_text("rcMSE")
    ax2.title.set_text("rpMSE")
    ax3.title.set_text("cpMSE")
    plt.show()

    # Abs and phase error histograms
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(12, 4.8)
    ax1.hist(errAbs, 100, density=False, facecolor='g', alpha=0.75)
    ax2.hist(errPhase, 100, density=False, facecolor='g', alpha=0.75)
    ax1.title.set_text("err_abs")
    ax2.title.set_text("err_phase")
    plt.show()




def compute_losses(y, yt, y_predict, type):
    # y  = RIR frequency domain
    # yt = RIR time domain
    # y_predict = RIR frequency domain prediction

    #print("y.shape = " + str(y.shape))
    #print("yt.shape = " + str(yt.shape))
    #print("y_predict.shape = " + str(y_predict.shape))
 
    n_rir_gt = y.shape[2]
    n_rir_pred = y_predict.shape[2]

    y_c = y[:,0,::n_rir_gt//n_rir_pred,:].float() + 1j*y[:,1,::n_rir_gt//n_rir_pred,:].float()
    y_hat_c = y_predict[:,0,:,:] + 1j*y_predict[:,1,:,:]

    err_abs = nn.functional.l1_loss(t.log(t.abs(y_hat_c)),t.log(t.abs(y_c)))    

    y1 = t.sin(t.angle(y_c))
    y2 = t.cos(t.angle(y_c))
    y_hat1 = t.sin(t.angle(y_hat_c))
    y_hat2 = t.cos(t.angle(y_hat_c))
    err_phase = nn.functional.l1_loss(y1,y_hat1) + nn.functional.l1_loss(y2,y_hat2)

    return err_abs, err_phase


if __name__=='__main__':
    args = p.parse_args()
    run_checkpoint(**vars(args))