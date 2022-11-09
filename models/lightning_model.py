from torch import optim, nn
import pytorch_lightning as pl
import torch as t
from utils.utils import getConfig
from WaveUnet import *

def getModel(model_name=None,learning_rate=1e-3):
    if   model_name == "SpeechDAREUnet_v1": model = SpeechDAREUnet_v1(learning_rate=learning_rate)
    elif model_name == "ErnstUnet":         model = ErnstUnet        (learning_rate=learning_rate)
    elif model_name == "Waveunet":          model = None # hack because I don't have the input params for the waveunet here
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

############################################################
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
        self.init()

    def init(self):
        k = 5
        s = 2
        # UNet model from "Speech Dereverberation Using Fully Convolutional Networks," Ernst et al., EUSIPCO 2018
        self.conv1 = nn.Sequential(nn.Conv2d(  1,  64, k, stride=s, padding=k//2), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d( 64, 128, k, stride=s, padding=k//2), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, k, stride=s, padding=k//2), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 256, k, stride=s, padding=k//2), nn.BatchNorm2d(256), nn.ReLU())
        
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(256, 256, k, stride=s, padding=k//2, output_padding=s-1), nn.BatchNorm2d(256), nn.Dropout2d(p=0.5), nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(512, 128, k, stride=s, padding=k//2, output_padding=s-1), nn.BatchNorm2d(128), nn.Dropout2d(p=0.5), nn.ReLU())
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(256,  64, k, stride=s, padding=k//2, output_padding=s-1), nn.BatchNorm2d(64),  nn.ReLU())
        self.deconv4 = nn.Sequential(nn.ConvTranspose2d(128,   1, k, stride=s, padding=k//2, output_padding=s-1), nn.Tanh())

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y, z = batch # reverberant speech, clean speech, RIR (RIR not used in this base UNet model)

        # batch_size x 1 (mag stft only) x 256 x 256
        x = x[:,[0],:,:].float()
        y = y[:,[0],:,:].float()

        y_hat = self.predict(x)
        loss   = nn.functional.mse_loss(y_hat, y)
        
        self.log("loss", {'train': loss })
        self.log("train_loss", loss )
        return loss

    def validation_step(self, batch, batch_idx):
        # validation_step defines the validation loop.
        # it is independent of forward
        x, y, z = batch # reverberant speech, clean speech, RIR (RIR not used in this base UNet model)
        x = x[:,[0],:,:].float()
        y = y[:,[0],:,:].float()
        
        y_hat = self.predict(x)
        loss   = nn.functional.mse_loss(y_hat, y)
        
        self.log("loss", {'val': loss })
        self.log("val_loss", loss )
        return loss

    def test_step(self, batch, batch_idx):
        # test_step defines the test loop.
        # it is independent of forward
        x, y, z = batch # reverberant speech, clean speech, RIR (RIR not used in this base UNet model)
        x = x[:,[0],:,:].float()
        y = y[:,[0],:,:].float()
        
        y_hat = self.predict(x)
        loss   = nn.functional.mse_loss(y_hat, y)
        
        self.log("loss", {'test': loss })
        self.log("test_loss", loss )
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

####################################################################

class SpeechDAREUnet_v2(pl.LightningModule):
    def __init__(self,learning_rate=1e-3):
        super().__init__()
        self.name = "SpeechDAREUnet_v1"

        self.has_init = False
        self.learning_rate = learning_rate
        self.init()

    def init(self):
        k = 5
        s = 2
        # UNet model from "Speech Dereverberation Using Fully Convolutional Networks," Ernst et al., EUSIPCO 2018
        self.conv1 = nn.Sequential(nn.Conv2d(  1,  64, k, stride=s, padding=k//2), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d( 64, 128, k, stride=s, padding=k//2), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, k, stride=s, padding=k//2), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 256, k, stride=s, padding=k//2), nn.BatchNorm2d(256), nn.ReLU())
        
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(256, 256, k, stride=s, padding=k//2, output_padding=s-1), nn.BatchNorm2d(256), nn.Dropout2d(p=0.5), nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(512, 128, k, stride=s, padding=k//2, output_padding=s-1), nn.BatchNorm2d(128), nn.Dropout2d(p=0.5), nn.ReLU())
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(256,  64, k, stride=s, padding=k//2, output_padding=s-1), nn.BatchNorm2d(64),  nn.ReLU())
        self.deconv4 = nn.Sequential(nn.ConvTranspose2d(128,   1, k, stride=s, padding=k//2, output_padding=s-1), nn.Tanh())

        # The idea for this simple RIR reconstruction branch is expand the 1 x 1 x 256 encoding into a 1 x 1 x 32000
        # RIR by increasing the number of channels but never the height or width.
        # The sizes for the RIR reconstruction assume the encoding is 1 x 1 x 256 as is labeled in predict(),
        # but with the current settings I think it's actually 16 x 16 x 256. Also, I'm not sure if BatchNorm2D and Dropout2d
        # are correct since the tensor is always 1 x 1 x N, i.e. not really 2d.
        # It make more more sense to reshape the tensor and use 1d operations instead.

        self.rir1 = nn.Sequential(nn.ConvTranspose2d(256,    1024, 1, stride=1, padding=0, output_padding=0), nn.BatchNorm2d(1024),  nn.Dropout2d(p=0.5), nn.ReLU())
        self.rir2 = nn.Sequential(nn.ConvTranspose2d(1024,   4096, 1, stride=1, padding=0, output_padding=0), nn.BatchNorm2d(4096),  nn.Dropout2d(p=0.5), nn.ReLU())
        self.rir3 = nn.Sequential(nn.ConvTranspose2d(4096,  16384, 1, stride=1, padding=0, output_padding=0), nn.BatchNorm2d(16384), nn.Dropout2d(p=0.5), nn.ReLU())
        self.rir4 = nn.Sequential(nn.ConvTranspose2d(16384, 32000, 1, stride=1, padding=0, output_padding=0), nn.BatchNorm2d(32000), nn.Dropout2d(p=0.5), nn.ReLU())


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y, z = batch # reverberant speech, clean speech, RIR (RIR not used in this base UNet model)

        x = x[:,[0],:,:].float()
        y = y[:,[0],:,:].float()

        y_hat = self.predict(x)
        loss   = nn.functional.mse_loss(y_hat, y)
        
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # validation_step defines the validation loop.
        # it is independent of forward
        x, y, z = batch # reverberant speech, clean speech, RIR (RIR not used in this base UNet model)
        x = x[:,[0],:,:].float()
        y = y[:,[0],:,:].float()
        
        y_hat = self.predict(x)
        loss   = nn.functional.mse_loss(y_hat, y)
        
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        # test_step defines the test loop.
        # it is independent of forward
        x, y, z = batch # reverberant speech, clean speech, RIR (RIR not used in this base UNet model)
        x = x[:,[0],:,:].float()
        y = y[:,[0],:,:].float()
        
        y_hat = self.predict(x)
        loss   = nn.functional.mse_loss(y_hat, y)
        
        self.log("test_loss", loss)
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
####################################################################
# IGNORE THIS: I will delete it once I get the real WaveUnet running
class Wave_UNet_v1(pl.LightningModule):
    def __init__(self,learning_rate=1e-3):
        super().__init__()
        self.name = "Wave_UNet_v1"

        self.has_init = False
        self.learning_rate = learning_rate
        self.init()
    
    
    def init(self):
        k = 5
        s = 2
        
        # For now, manually fix the number of layers at 12. Eventually make this a parameter and create downsample and upsample blocks
        # Start with 2 seconds of data @ 16 kHz so the sizes might be (if reducing by 2x): 32k, 16k, 8k, 4k, 2k, 1k, 500, 250, 125
        # Start with 32768 samples  of data @ 16 kHz (~2.05s) so the sizes go down to 4 samples ans 12*24 = 288 channels
        Fc = 24 # extra filters per layer
        fd = 15 # downsampling conv size
        fu = 5  # upsampling conv size
        pd = 'same' # padding for downsampling so the conv output is the same size as the input
        s  = 2
        m  = 0.01 # Wave-U-Net uses momentum 0.01 for the BatchNorm1d layers, the default is 0.1
        
        self.down1  = nn.Sequential(nn.Conv1d(   1,  Fc*1,  fd, padding=pd, stride=s), nn.BatchNorm1d(Fc*1,  m), nn.LeakyReLU(0.2))
        self.down2  = nn.Sequential(nn.Conv1d(Fc*1,  Fc*2,  fd, padding=pd, stride=s), nn.BatchNorm1d(Fc*2,  m), nn.LeakyReLU(0.2))
        self.down3  = nn.Sequential(nn.Conv1d(Fc*2,  Fc*3,  fd, padding=pd, stride=s), nn.BatchNorm1d(Fc*3,  m), nn.LeakyReLU(0.2))
        self.down4  = nn.Sequential(nn.Conv1d(Fc*3,  Fc*4,  fd, padding=pd, stride=s), nn.BatchNorm1d(Fc*4,  m), nn.LeakyReLU(0.2))
        self.down5  = nn.Sequential(nn.Conv1d(Fc*4,  Fc*5,  fd, padding=pd, stride=s), nn.BatchNorm1d(Fc*5,  m), nn.LeakyReLU(0.2))
        self.down6  = nn.Sequential(nn.Conv1d(Fc*5,  Fc*6,  fd, padding=pd, stride=s), nn.BatchNorm1d(Fc*6,  m), nn.LeakyReLU(0.2))
        self.down7  = nn.Sequential(nn.Conv1d(Fc*6,  Fc*7,  fd, padding=pd, stride=s), nn.BatchNorm1d(Fc*7,  m), nn.LeakyReLU(0.2))
        self.down8  = nn.Sequential(nn.Conv1d(Fc*7,  Fc*8,  fd, padding=pd, stride=s), nn.BatchNorm1d(Fc*8,  m), nn.LeakyReLU(0.2))
        self.down9  = nn.Sequential(nn.Conv1d(Fc*8,  Fc*9,  fd, padding=pd, stride=s), nn.BatchNorm1d(Fc*9,  m), nn.LeakyReLU(0.2))
        self.down10 = nn.Sequential(nn.Conv1d(Fc*9,  Fc*10, fd, padding=pd, stride=s), nn.BatchNorm1d(F*10,  m), nn.LeakyReLU(0.2))
        self.down11 = nn.Sequential(nn.Conv1d(Fc*10, Fc*11, fd, padding=pd, stride=s), nn.BatchNorm1d(Fc*11, m), nn.LeakyReLU(0.2))
        self.down12 = nn.Sequential(nn.Conv1d(Fc*11, Fc*12, fd, padding=pd, stride=s), nn.BatchNorm1d(Fc*12, m), nn.LeakyReLU(0.2))

        
       

