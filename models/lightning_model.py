from cmath import nan
import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
import torch as t

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
        self.init()

    def init(self):
        k = 5
        s = 2
        # UNet model from "Speech Dereverberation Using Fully Convolutional Networks," Ernst et al., EUSIPCO 2018
        self.add_module("conv1", nn.Sequential(nn.Conv2d(  1,  64, k, stride=s, padding=k//2), nn.LeakyReLU(0.2)))
        self.add_module("conv2", nn.Sequential(nn.Conv2d( 64, 128, k, stride=s, padding=k//2), nn.BatchNorm2d(128), nn.LeakyReLU(0.2)))
        self.add_module("conv3", nn.Sequential(nn.Conv2d(128, 256, k, stride=s, padding=k//2), nn.BatchNorm2d(256), nn.LeakyReLU(0.2)))
        self.add_module("conv4", nn.Sequential(nn.Conv2d(256, 256, k, stride=s, padding=k//2), nn.BatchNorm2d(256), nn.ReLU()))
        
        self.add_module("deconv1", nn.Sequential(nn.ConvTranspose2d(256, 256, k, stride=s, padding=k//2, output_padding=s-1), nn.BatchNorm2d(256), nn.Dropout2d(p=0.5), nn.ReLU()))
        self.add_module("deconv2", nn.Sequential(nn.ConvTranspose2d(512, 128, k, stride=s, padding=k//2, output_padding=s-1), nn.BatchNorm2d(128), nn.Dropout2d(p=0.5), nn.ReLU()))
        self.add_module("deconv3", nn.Sequential(nn.ConvTranspose2d(256,  64, k, stride=s, padding=k//2, output_padding=s-1), nn.BatchNorm2d(64),  nn.ReLU()))
        self.add_module("deconv4", nn.Sequential(nn.ConvTranspose2d(128,   1, k, stride=s, padding=k//2, output_padding=s-1), nn.Tanh()))
    
        #a = self.conv1.state_dict()
        #a['0.weight'][0][0][0] 
        #print(a['0.weight'][0][0][0])
        #print(a['0.weight'].device)

    def training_step(self, batch, batch_idx):
        #if not self.has_init:
            #self.init(learning_rate=self.learning_rate)
            #self.to(batch[0].device.type)
            #self.has_init = True
        
        # training_step defines the train loop.
        # it is independent of forward
        x, y, z = batch # reverberant speech, clean speech, RIR (RIR not used in this base UNet model)
        x = x[:,[0],:,:]
        y = y[:,[0],:,:]

        #if t.all(x == 0):
        #    print("WTF!?!?!?")
            #x = t.randn(x.shape).to(x.device)

        #if t.all(y == 0):
        #    print("WTF AGAIN!?!?!?")
            #y = t.randn(y.shape).to(y.device)

        #if self.conv1.state_dict()['0.weight'][0][0][0][0] == 0:
            #print("zeros again!?")
            
            #self.init()
            #self.to('cuda')
        
            #self.train(True)
            #self.unfreeze()



        #print(self.conv1.state_dict()['0.weight'][0][0][0])
        #print(self.conv2.state_dict()['0.weight'][0][0][0])

        y_hat = self.predict(x)

        #print(x[0,0,100:105,100])
        #print(y[0,0,100:105,100])
        #print(y_hat[0,0,100:105,100])        

        loss   = nn.functional.mse_loss(y_hat, y)
        
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        #if not self.has_init:
            #self.init()
            #self.to(batch[0].device.type)
            #self.has_init = True
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