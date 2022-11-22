from torch import optim, nn
from torch.optim import lr_scheduler
import pytorch_lightning as pl
import torch as t
from torchmetrics import ScaleInvariantSignalDistortionRatio
import matplotlib.pyplot as plt
import numpy as np

def getModel(model_name=None,learning_rate=1e-3,nfft=511,nhop=256):
    if   model_name == "SpeechDAREUnet_v1": model = SpeechDAREUnet_v1(learning_rate=learning_rate)
    elif model_name == "SpeechDAREUnet_v2": model = SpeechDAREUnet_v2(learning_rate=learning_rate,nfft=nfft,nhop=nhop)
    elif model_name == "ErnstUnet":         model = ErnstUnet        (learning_rate=learning_rate)
    else: raise Exception("Unknown model name.")
    
    return model

# define the LightningModule
class LitAutoEncoder(pl.LightningModule):
    def __init__(self,learning_rate=1e-3):
        super().__init__()
        self.name = "LitAutoEncoder"

        self.learning_rate = learning_rate

        self.encoder = nn.Sequential(nn.Linear(256*256*2, 64), nn.ReLU(), nn.Linear(64, 8))
        self.decoder = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 256*256*2))

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y, z = batch # reverberant speech, clean speech, RIR
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)
        z = z.view(z.size(0), -1)
        latent = self.encoder(x)
        y_hat = self.decoder(latent)
        loss = nn.functional.mse_loss(y_hat, y)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # validation_step defines the validation loop.
        # it is independent of forward
        x, y, z = batch # reverberant speech, clean speech, RIR
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)
        z = z.view(z.size(0), -1)
        latent = self.encoder(x)
        y_hat = self.decoder(latent)
        loss = nn.functional.mse_loss(y_hat, y)
        # Logging to TensorBoard by default
        self.log("val_loss", loss)
        return loss

    #def test_step(self, batch, batch_idx):
        # test_step defines the test loop.
        #     
           
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def predict(self, x):
        x = x.view(x.size(0), -1)
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        return x_hat

class ErnstUnet(pl.LightningModule):
    def __init__(self,learning_rate=1e-3):
        super().__init__()
        self.name = "ErnstUnet"

        self.learning_rate = learning_rate

        # UNet model from "Speech Dereverberation Using Fully Convolutional Networks," Ernst et al., EUSIPCO 2018
        self.conv1   = nn.Sequential(nn.Conv2d(1,    64, 5, stride=2, padding=2), nn.LeakyReLU(0.2))
        self.conv2   = nn.Sequential(nn.Conv2d(64,  128, 5, stride=2, padding=2), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.conv3   = nn.Sequential(nn.Conv2d(128, 256, 5, stride=2, padding=2), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))
        self.conv4   = nn.Sequential(nn.Conv2d(256, 512, 5, stride=2, padding=2), nn.BatchNorm2d(512), nn.LeakyReLU(0.2))
        self.conv5   = nn.Sequential(nn.Conv2d(512, 512, 5, stride=2, padding=2), nn.BatchNorm2d(512), nn.LeakyReLU(0.2))
        self.conv6   = nn.Sequential(nn.Conv2d(512, 512, 5, stride=2, padding=2), nn.BatchNorm2d(512), nn.LeakyReLU(0.2))
        self.conv7   = nn.Sequential(nn.Conv2d(512, 512, 5, stride=2, padding=2), nn.BatchNorm2d(512), nn.LeakyReLU(0.2))
        self.conv8   = nn.Sequential(nn.Conv2d(512, 512, 5, stride=2, padding=2), nn.BatchNorm2d(512), nn.ReLU())
        
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d( 512, 512, 5, stride=2, padding=2, output_padding=1), nn.BatchNorm2d(512), nn.Dropout2d(p=0.5), nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(1024, 512, 5, stride=2, padding=2, output_padding=1), nn.BatchNorm2d(512), nn.Dropout2d(p=0.5), nn.ReLU())
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(1024, 512, 5, stride=2, padding=2, output_padding=1), nn.BatchNorm2d(512), nn.Dropout2d(p=0.5), nn.ReLU())
        self.deconv4 = nn.Sequential(nn.ConvTranspose2d(1024, 512, 5, stride=2, padding=2, output_padding=1), nn.BatchNorm2d(512), nn.ReLU())
        self.deconv5 = nn.Sequential(nn.ConvTranspose2d(1024, 256, 5, stride=2, padding=2, output_padding=1), nn.BatchNorm2d(256),  nn.ReLU())
        self.deconv6 = nn.Sequential(nn.ConvTranspose2d( 512, 128, 5, stride=2, padding=2, output_padding=1), nn.BatchNorm2d(128),  nn.ReLU())
        self.deconv7 = nn.Sequential(nn.ConvTranspose2d( 256,  64, 5, stride=2, padding=2, output_padding=1), nn.BatchNorm2d(64),  nn.ReLU())
        self.deconv8 = nn.Sequential(nn.ConvTranspose2d( 128,   1, 5, stride=2, padding=2, output_padding=1), nn.Tanh())
    

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y, z = batch # reverberant speech, clean speech, RIR (RIR not used in this base UNet model)
        x = x[:,[0],:,:]
        y = y[:,[0],:,:]
        
        y_hat = self.predict(x)
        loss   = nn.functional.mse_loss(y_hat, y)
        
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # validation_step defines the validation loop.
        # it is independent of forward
        x, y, z = batch # reverberant speech, clean speech, RIR (RIR not used in this base UNet model)
        x = x[:,[0],:,:]
        y = y[:,[0],:,:]
        
        y_hat = self.predict(x)
        loss   = nn.functional.mse_loss(y_hat, y)
        
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        # test_step defines the test loop.
        # it is independent of forward
        x, y, z = batch # reverberant speech, clean speech, RIR (RIR not used in this base UNet model)
        x = x[:,[0],:,:]
        y = y[:,[0],:,:]
        
        y_hat = self.predict(x)
        loss   = nn.functional.mse_loss(y_hat, y)
        
        self.log("test_loss", loss)
        return loss

    def predict(self, x):
        c1Out = self.conv1(x)     # (128 x 128 x 64)
        c2Out = self.conv2(c1Out) # (64 x 64 x 128)
        c3Out = self.conv3(c2Out) # (32 x 32 x 256)
        c4Out = self.conv4(c3Out) # (16 x 16 x 512)
        c5Out = self.conv5(c4Out) # (8 x 8 x 512)
        c6Out = self.conv6(c5Out) # (4 x 4 x 512)
        c7Out = self.conv7(c6Out) # (2 x 2 x 512)
        c8Out = self.conv8(c7Out) # (1 x 1 x 512)

        d1Out = self.deconv1(c8Out) # (2 x 2 x 1024)
        d2Out = self.deconv2(t.cat((d1Out, c7Out), dim=1)) # (4 x 4 x 1024)
        d3Out = self.deconv3(t.cat((d2Out, c6Out), dim=1)) # (8 x 8 x 1024)
        d4Out = self.deconv4(t.cat((d3Out, c5Out), dim=1)) # (16 x 16 x 1024)
        d5Out = self.deconv5(t.cat((d4Out, c4Out), dim=1)) # (32 x 32 x 512)
        d6Out = self.deconv6(t.cat((d5Out, c3Out), dim=1)) # (64 x 64 x 256)
        d7Out = self.deconv7(t.cat((d6Out, c2Out), dim=1)) # (128 x 128 x 128)
        d8Out = self.deconv8(t.cat((d7Out, c1Out), dim=1)) # (256 x 256 x 1)
        return d8Out

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

class SpeechDAREUnet_v1(pl.LightningModule):
    def __init__(self,learning_rate=1e-3):
        super().__init__()
        self.name = "SpeechDAREUnet_v1"

        self.has_init = False
        self.learning_rate = learning_rate
        self.si_sdr = ScaleInvariantSignalDistortionRatio()
        self.nfft = 511
        self.nhop = 256
        self.win = t.hann_window(self.nfft)
        self.init()

    def init(self):
        k = 5
        s = 2
        # Small UNet model
        self.conv1 = nn.Sequential(nn.Conv2d(  2,  64, k, stride=s, padding=k//2), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d( 64, 128, k, stride=s, padding=k//2), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, k, stride=s, padding=k//2), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 256, k, stride=s, padding=k//2), nn.BatchNorm2d(256), nn.ReLU())
        
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(256, 256, k, stride=s, padding=k//2, output_padding=s-1), nn.BatchNorm2d(256), nn.Dropout2d(p=0.5), nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(512, 128, k, stride=s, padding=k//2, output_padding=s-1), nn.BatchNorm2d(128), nn.Dropout2d(p=0.5), nn.ReLU())
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(256,  64, k, stride=s, padding=k//2, output_padding=s-1), nn.BatchNorm2d(64),  nn.ReLU())
        self.deconv4 = nn.Sequential(nn.ConvTranspose2d(128,   2, k, stride=s, padding=k//2, output_padding=s-1), nn.Tanh())

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y, z, rir = batch # reverberant speech, clean speech, RIR (RIR not used in this base UNet model)
        
        x = x.float()
        y = y.float()
        y_hat = self.predict(x.float())

        y_c = y[:,0,:,:].float() + 1j*y[:,1,:,:].float()
        y_hat_c = y_hat[:,0,:,:] + 1j*y_hat[:,1,:,:]
        mse_abs = nn.functional.mse_loss(
            t.log(t.abs(y_hat_c)),
            t.log(t.abs(y_c))
            )
        phasedist = t.sum(t.abs(y_c)/t.sum(t.abs(y_c)) * t.abs(t.angle(y_c)-t.angle(y_hat_c)))
        # loss = mse_abs + phasedist

        y_hat_wav = t.istft(y_hat_c,self.nfft,hop_length=self.nhop,win_length=self.nfft,window=self.win.to(self.device),length=z.shape[1])
        loss = - self.si_sdr(y_hat_wav, z) + mse_abs*10

        # Try real then imag and loss=real+imag+abs 
        # xr = x[:,[0],:,:].float()
        # xi = x[:,[1],:,:].float()
        # yr = y[:,[0],:,:].float()
        # yi = y[:,[1],:,:].float()
        # yr_hat = self.predict(xr)
        # yi_hat = self.predict(xi)
        # loss_real = nn.functional.mse_loss(yr_hat, yr)
        # loss_imag = nn.functional.mse_loss(yi_hat, yi)
        # loss_mag  = nn.functional.mse_loss(t.log(t.abs(yr_hat + 1j*yi_hat)), t.log(t.abs(yr + 1j*yi)))
        # loss = loss_real + loss_imag + 2*loss_mag
        
        self.log("loss", {'train': loss })
        self.log("train_loss", loss )
        self.log("train_phasedist", phasedist )
        self.log("train_mse_abs", mse_abs )
        return loss

    def validation_step(self, batch, batch_idx):
        # validation_step defines the validation loop.
        # it is independent of forward
        x, y, z, rir = batch # reverberant speech, clean speech, RIR (RIR not used in this base UNet model)
        
        x = x.float()
        y = y.float()
        y_hat = self.predict(x.float())

        y_c = y[:,0,:,:].float() + 1j*y[:,1,:,:].float()
        y_hat_c = y_hat[:,0,:,:] + 1j*y_hat[:,1,:,:]
        mse_abs = nn.functional.mse_loss(
            t.log(t.abs(y_hat_c)),
            t.log(t.abs(y_c))
            )
        phasedist = t.sum(t.abs(y_c)/t.sum(t.abs(y_c)) * t.abs(t.angle(y_c)-t.angle(y_hat_c)))
        # loss = mse_abs + phasedist

        y_hat_wav = t.istft(y_hat_c,self.nfft,hop_length=self.nhop,win_length=self.nfft,window=self.win.to(self.device),length=z.shape[1])
        loss = - self.si_sdr(y_hat_wav, z) + mse_abs*10

        # Try real then imag and loss=real+imag+abs 
        # xr = x[:,[0],:,:].float()
        # xi = x[:,[1],:,:].float()
        # yr = y[:,[0],:,:].float()
        # yi = y[:,[1],:,:].float()
        # yr_hat = self.predict(xr)
        # yi_hat = self.predict(xi)
        # loss_real = nn.functional.mse_loss(yr_hat, yr)
        # loss_imag = nn.functional.mse_loss(yi_hat, yi)
        # loss_mag  = nn.functional.mse_loss(t.log(t.abs(yr_hat + 1j*yi_hat)), t.log(t.abs(yr + 1j*yi)))
        # loss = loss_real + loss_imag + 2*loss_mag
        
        if (self.current_epoch % 1 == 0) and (batch_idx==0):
            import matplotlib.pyplot as plt
            import numpy as np
            fh = plt.figure()
            fig, ((ax1, ax2, ax3),(ax4, ax5, ax6),(ax7, ax8, ax9)) = plt.subplots(3, 3)
            a = x.cpu().squeeze().detach().numpy()
            b = y.cpu().squeeze().detach().numpy()
            c = y_hat.cpu().squeeze().detach().numpy()
            ax1.imshow(a[0,0,:,:])
            ax2.imshow(b[0,0,:,:])
            ax3.imshow(c[0,0,:,:])
            ax4.imshow(a[0,1,:,:])
            ax5.imshow(b[0,1,:,:])
            ax6.imshow(c[0,1,:,:])
            ax7.imshow(np.log(np.abs(a[0,0,:,:] + 1j*a[0,1,:,:])))
            ax8.imshow(np.log(np.abs(b[0,0,:,:] + 1j*b[0,1,:,:])))
            ax9.imshow(np.log(np.abs(c[0,0,:,:] + 1j*c[0,1,:,:])))
            plt.savefig("./images/2_"+str(self.current_epoch)+".png",dpi=600)
            plt.close()


        self.log("loss", {'val': loss })
        self.log("val_loss", loss )
        self.log("val_phasedist", phasedist )
        self.log("val_mse_abs", mse_abs )
        return loss

    def test_step(self, batch, batch_idx):
        # test_step defines the test loop.
        # it is independent of forward
        x, y, z, rir = batch # reverberant speech, clean speech, RIR (RIR not used in this base UNet model)
        
        x = x.float()
        y = y.float()
        y_hat = self.predict(x.float())

        y_c = y[:,0,:,:].float() + 1j*y[:,1,:,:].float()
        y_hat_c = y_hat[:,0,:,:] + 1j*y_hat[:,1,:,:]
        mse_abs = nn.functional.mse_loss(
            t.log(t.abs(y_hat_c)),
            t.log(t.abs(y_c))
            )
        phasedist = t.sum(t.abs(y_c)/t.sum(t.abs(y_c)) * t.abs(t.angle(y_c)-t.angle(y_hat_c)))
        # loss = mse_abs + phasedist

        y_hat_wav = t.istft(y_hat_c,self.nfft,hop_length=self.nhop,win_length=self.nfft,window=self.win.to(self.device),length=z.shape[1])
        loss = - self.si_sdr(y_hat_wav, z) + mse_abs*10

        # Try real then imag and loss=real+imag+abs 
        # xr = x[:,[0],:,:].float()
        # xi = x[:,[1],:,:].float()
        # yr = y[:,[0],:,:].float()
        # yi = y[:,[1],:,:].float()
        # yr_hat = self.predict(xr)
        # yi_hat = self.predict(xi)
        # loss_real = nn.functional.mse_loss(yr_hat, yr)
        # loss_imag = nn.functional.mse_loss(yi_hat, yi)
        # loss_mag  = nn.functional.mse_loss(t.log(t.abs(yr_hat + 1j*yi_hat)), t.log(t.abs(yr + 1j*yi)))
        # loss = loss_real + loss_imag + 2*loss_mag
        
        self.log("loss", {'test': loss })
        self.log("test_loss", loss )
        self.log("test_phasedist", phasedist )
        self.log("test_mse_abs", mse_abs )
        return loss

    def predict(self, x):
        c1Out = self.conv1(x)     # (64 x 64 x  64)
        c2Out = self.conv2(c1Out) # (16 x 16 x 128)
        c3Out = self.conv3(c2Out) # ( 4 x  4 x 256)
        c4Out = self.conv4(c3Out) # ( 1 x  1 x 256)

        d1Out = self.deconv1(c4Out) # (  4 x   4 x 256)
        d2Out = self.deconv2(t.cat((d1Out, c3Out), dim=1)) # ( 16 x  16 x 128)
        d3Out = self.deconv3(t.cat((d2Out, c2Out), dim=1)) # ( 64 x  64 x 128)
        d4Out = self.deconv4(t.cat((d3Out, c1Out), dim=1)) # (256 x 256 x 1)
        return d4Out

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer



class SpeechDAREUnet_v2(pl.LightningModule):
    def __init__(self,
        learning_rate=1e-3,
        nfft=2**15-1,
        nhop=(2**15)/(2**6),
        nfrms=16):
        super().__init__()
        self.name = "SpeechDAREUnet_v2"

        self.has_init = False
        self.learning_rate = learning_rate
        self.lr_scheduler_gamma = 0.9
        self.loss_ind = 0
        self.si_sdr = ScaleInvariantSignalDistortionRatio()
        self.nfft = nfft
        self.nhop = nhop
        self.nfrms = nfrms
        self.win = t.hann_window(self.nfft)
        self.eps = 1e-16
        self.init()

    def init(self):
        k = 5
        s = 2
        p_drop = 0.5
        leaky_slope = 0.01
        # Small UNet model
        self.conv1 = nn.Sequential(nn.Conv2d(  2,  64, k, stride=s, padding=k//2), nn.LeakyReLU(leaky_slope))
        self.conv2 = nn.Sequential(nn.Conv2d( 64, 128, k, stride=s, padding=k//2), nn.BatchNorm2d(128), nn.LeakyReLU(leaky_slope))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, k, stride=s, padding=k//2), nn.BatchNorm2d(256), nn.LeakyReLU(leaky_slope))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 256, k, stride=s, padding=k//2), nn.BatchNorm2d(256), nn.ReLU())
        
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(256, 256, k, stride=s, padding=k//2, output_padding=s-1), nn.BatchNorm2d(256), nn.Dropout2d(p=p_drop), nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(512, 128, k, stride=s, padding=k//2, output_padding=s-1), nn.BatchNorm2d(128), nn.Dropout2d(p=p_drop), nn.ReLU())
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(256,  64, k, stride=s, padding=k//2, output_padding=s-1), nn.BatchNorm2d(64),  nn.ReLU())
        self.deconv4 = nn.Sequential(nn.ConvTranspose2d(128,   2, k, stride=s, padding=k//2, output_padding=s-1), nn.BatchNorm2d(2),  nn.ReLU())
        
        self.out1 = nn.Sequential(nn.Conv2d(2,   2, (1,self.nfrms), stride=1, padding=0), nn.Tanh())

    def training_step(self, batch, batch_idx):
        loss_type = "train"
        # training_step defines the train loop.
        # it is independent of forward
        x, _, _, y, yt = batch # reverberant speech, clean speech, waveform speech, RIR fft
        x = x.float()
        y = y.float()
        y_hat = self.predict(x.float())

        loss = self.compute_losses(y, yt, y_hat, loss_type)[self.loss_ind]
        self.log("loss", {loss_type: loss })
        self.log(loss_type+"_loss", loss )
        return loss

    def validation_step(self, batch, batch_idx):
        loss_type = "val"
        # validation_step defines the validation loop.
        # it is independent of forward
        x, _, _, y, yt = batch # reverberant speech, clean speech, waveform speech, RIR fft
        x = x.float()
        y = y.float()
        y_hat = self.predict(x.float())
        
        losses = self.compute_losses(y, yt, y_hat, loss_type)
        loss = losses[self.loss_ind]
        self.log("loss", {loss_type: loss })
        self.log(loss_type+"_loss", loss )
        
        self.make_plot(batch_idx, x, y, y_hat, losses[-1])

        self.weight_histograms()

        return loss

    def test_step(self, batch, batch_idx):
        loss_type = "test"
        # test_step defines the test loop.
        # it is independent of forward
        x, _, _, y, yt = batch # reverberant speech, clean speech, waveform speech, RIR fft
        x = x.float()
        y = y.float()
        y_hat = self.predict(x.float())

        loss = self.compute_losses(y, yt, y_hat, loss_type)[self.loss_ind]
        self.log("loss", {loss_type: loss })
        self.log(loss_type+"_loss", loss )
        return loss

    def predict(self, x):
        c1Out = self.conv1(x)     # (64 x 64 x  64)
        c2Out = self.conv2(c1Out) # (16 x 16 x 128)
        c3Out = self.conv3(c2Out) # ( 4 x  4 x 256)
        c4Out = self.conv4(c3Out) # ( 1 x  1 x 256)

        d1Out = self.deconv1(c4Out) # (  4 x   4 x 256)
        d2Out = self.deconv2(t.cat((d1Out, c3Out), dim=1)) # ( 16 x  16 x 128)
        d3Out = self.deconv3(t.cat((d2Out, c2Out), dim=1)) # ( 64 x  64 x 128)
        d4Out = self.deconv4(t.cat((d3Out, c1Out), dim=1)) # (256 x 256 x 1)
        out1Out = self.out1(d4Out) # (256 x 256 x 1)
        return out1Out
    
    def compute_losses(self, y, yt, y_predict, type):
        y_c = y[:,0,:,:].float() + 1j*y[:,1,:,:].float()
        y_hat_c = y_predict[:,0,:,:] + 1j*y_predict[:,1,:,:]
        
        y_hat_c_t = t.fft.irfft(y_hat_c,n=yt.shape[1],dim=1).squeeze()
        mse_time = nn.functional.mse_loss(y_hat_c_t, yt)
        err_time = nn.functional.l1_loss(y_hat_c_t, yt)
        y_hat_c_t_abs_log = (y_hat_c_t.abs()+self.eps).log()
        yt_abs_log = (yt.abs()+self.eps).log()
        mse_time_abs_log = nn.functional.mse_loss(y_hat_c_t_abs_log, yt_abs_log)
        err_time_abs_log = nn.functional.l1_loss(y_hat_c_t_abs_log, yt_abs_log)
        kld_time_abs_log = nn.functional.kl_div(y_hat_c_t_abs_log,yt_abs_log,log_target=True).abs()
        
        mse_real = nn.functional.mse_loss(t.real(y_hat_c),t.real(y_c))
        mse_imag = nn.functional.mse_loss(t.imag(y_hat_c),t.imag(y_c))
        mse_abs = nn.functional.mse_loss(t.log(t.abs(y_hat_c)),t.log(t.abs(y_c)))
        
        err_real = nn.functional.l1_loss(t.real(y_hat_c),t.real(y_c))
        err_imag = nn.functional.l1_loss(t.imag(y_hat_c),t.imag(y_c))
        err_abs = nn.functional.l1_loss(t.log(t.abs(y_hat_c)),t.log(t.abs(y_c)))
        
        y1 = t.sin(t.angle(y_c))
        y2 = t.cos(t.angle(y_c))
        y_hat1 = t.sin(t.angle(y_hat_c))
        y_hat2 = t.cos(t.angle(y_hat_c))
        # err_phase = t.sum(t.abs(y_c)/t.sum(t.abs(y_c)) * t.abs(t.angle(y_c)-t.angle(y_hat_c)))
        mse_phase = nn.functional.mse_loss(y1,y_hat1) + nn.functional.mse_loss(y2,y_hat2)
        err_phase = nn.functional.l1_loss(y1,y_hat1) + nn.functional.l1_loss(y2,y_hat2)
        y_a = t.tensor(np.unwrap(t.angle(y_c).cpu().detach().numpy(),axis=1)).to(self.device)
        y_hat_a = t.tensor(np.unwrap(t.angle(y_hat_c).cpu().detach().numpy(),axis=1)).to(self.device)
        mse_phase_un = nn.functional.mse_loss(y_a,y_hat_a)
        err_phase_un = nn.functional.l1_loss(y_a,y_hat_a)

        #loss = err_real + err_imag + 2*err_abs
        loss_err = err_abs + err_phase + err_phase_un * 1e-4
        loss_mse = mse_abs + mse_phase + mse_phase_un * 1e-4

        self.log(type+"_loss_err", loss_err )
        self.log(type+"_loss_mse", loss_mse )
        self.log(type+"_err_phase",err_phase)
        self.log(type+"_err_phase_un",err_phase_un)
        self.log(type+"_err_abs",  err_abs  )
        self.log(type+"_err_real", err_real )
        self.log(type+"_err_imag", err_imag )
        self.log(type+"_err_time", err_time )
        self.log(type+"_err_time_abs_log", err_time_abs_log )
        self.log(type+"_mse_phase",mse_phase)
        self.log(type+"_mse_phase_un",mse_phase_un)
        self.log(type+"_mse_abs",  mse_abs  )
        self.log(type+"_mse_real", mse_real )
        self.log(type+"_mse_imag", mse_imag )
        self.log(type+"_mse_time", mse_time )
        self.log(type+"_mse_time_abs_log", mse_time_abs_log )
        self.log(type+"_kld_time_abs_log", kld_time_abs_log )

        return \
            loss_err, loss_mse, \
            err_real, err_imag, err_abs, err_phase, err_phase_un, err_time, err_time_abs_log, \
            mse_real, mse_imag, mse_abs, mse_phase, mse_phase_un, mse_time, mse_time_abs_log, \
            kld_time_abs_log, \
            y_hat_c

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = lr_scheduler.ExponentialLR(optimizer, self.lr_scheduler_gamma, self.current_epoch-1)
        return [optimizer], [scheduler]

    def make_plot(self,batch_idx,x,y,y_hat,y_hat_c):
        if (self.current_epoch % 1 == 0) and (batch_idx==0):
            fh = plt.figure()
            fig, ((ax1, ax2, ax3),(ax4, ax5, ax6),(ax7, ax8, ax9),(ax10,ax11,ax12)) = plt.subplots(4, 3)
            a = x.cpu().detach().numpy()
            b = y.cpu().detach().numpy()
            c = y_hat.cpu().detach().numpy()
            
            y_c = y[0,0,:,:].float() + 1j*y[0,1,:,:].float()
            x_a = np.unwrap(np.angle((a[0,0,:,:] + 1j*a[0,1,:,:])),axis=0)
            y_a = np.unwrap(np.angle(y_c.cpu().detach().numpy()),axis=0)
            y_hat_a = np.unwrap(np.angle(y_hat_c.cpu().detach().numpy()),axis=1)
            ax1.imshow(a[0,0,:,:], aspect='auto')
            ax2.imshow(b[0,0,:,:], aspect='auto')
            ax3.imshow(c[0,0,:,:], aspect='auto')
            ax4.plot(x_a[:,0], linewidth=0.1)
            ax5.plot(y_a[:,0], linewidth=0.1)
            ax6.plot(y_hat_a[0,:,0], linewidth=0.1)
            # ax4.imshow(np.angle(a[0,0,:,:] + 1j*a[0,1,:,:]), aspect='auto')
            # ax5.imshow(np.angle(b[0,0,:,:] + 1j*b[0,1,:,:]), aspect='auto')
            # ax6.imshow(np.angle(c[0,0,:,:] + 1j*c[0,1,:,:]), aspect='auto')
            ax7.plot(np.mean(np.log(np.abs(a[0,0,:,:] + 1j*a[0,1,:,:])),axis=1), linewidth=0.1)
            ax8.plot(np.log(np.abs(b[0,0,:,0] + 1j*b[0,1,:,0])), linewidth=0.1)
            ax9.plot(np.log(np.abs(c[0,0,:,0] + 1j*c[0,1,:,0])), linewidth=0.1)
            rir_est = np.fft.irfft(y_hat_c[0,:,:].cpu().squeeze().detach().numpy())
            rir = np.fft.irfft(y_c.cpu().squeeze().detach().numpy())
            gs = fig.add_gridspec(4,3)
            ax10 = fig.add_subplot(gs[3, 0])
            ax10.plot(rir_est[:rir_est.shape[0]//16], linewidth=0.1)
            ax10.plot(rir[:rir_est.shape[0]//16], linewidth=0.1)
            ax11 = fig.add_subplot(gs[3, 1])
            ax11.plot(rir_est, linewidth=0.1)
            ax11.plot(rir, linewidth=0.1)
            ax12 = fig.add_subplot(gs[3, 2])
            ax12.plot(np.log(np.abs(rir_est)), linewidth=0.1)
            ax12.plot(np.log(np.abs(rir)), linewidth=0.1)
            plt.savefig("./images/2_"+str(self.current_epoch)+".png",dpi=1200)
            tb = self.logger.experiment
            tb.add_figure('Fig1', plt.gcf(), global_step=self.global_step)
            plt.close()
        return 0
    
    def weight_histograms_conv2d(self, writer, step, weights, layer_number):
        weights_shape = weights.shape
        num_kernels = weights_shape[0]
        for k in range(num_kernels):
            flattened_weights = weights[k].flatten()
            tag = f"layer_{layer_number}/kernel_{k}"
            writer.add_histogram(tag, flattened_weights, global_step=step, bins='tensorflow')

    def weight_histograms(self):
        writer = self.logger.experiment
        step = self.global_step
        # Iterate over all model layers
        layers = []
        layers.append(self.conv1[0])
        layers.append(self.conv2[0])
        layers.append(self.conv3[0])
        layers.append(self.conv4[0])
        layers.append(self.deconv1[0])
        layers.append(self.deconv2[0])
        layers.append(self.deconv3[0])
        layers.append(self.deconv4[0])
        layers.append(self.out1[0])
        
        for layer_number in range(len(layers)):
            layer = layers[layer_number]
            # Compute weight histograms for appropriate layer
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                weights = layer.weight
                self.weight_histograms_conv2d(writer, step, weights, layer_number)

