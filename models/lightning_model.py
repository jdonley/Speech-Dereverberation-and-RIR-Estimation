from torch import optim, nn
import pytorch_lightning as pl
import torch as t
from torchmetrics import ScaleInvariantSignalDistortionRatio

def getModel(model_name=None,learning_rate=1e-3):
    if   model_name == "SpeechDAREUnet_v1": model = SpeechDAREUnet_v1(learning_rate=learning_rate)
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
        loss = - self.si_sdr(y_hat_wav, z)

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
        loss = - self.si_sdr(y_hat_wav, z)

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
        
        if self.current_epoch % 1 == 0:
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
            plt.savefig("./images/"+str(self.current_epoch)+".png",dpi=600)
            plt.clf()


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
        loss = - self.si_sdr(y_hat_wav, z)

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