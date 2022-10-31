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

        # each x = (256 x 256 x 1)
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

        loss   = nn.functional.mse_loss(d8Out, y)
        #if batch_idx==0 or batch_idx==1000:
        import matplotlib.pyplot as plt
        if batch_idx == 9:
        #    for i in range(2):
        #        plt.imshow(x[i,0,:,:].squeeze().to('cpu').numpy())
        #        plt.show()
        #        plt.imshow(y[i,0,:,:].squeeze().to('cpu').numpy())
        #        plt.show()
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ax1.imshow(x[0,0,:,:].squeeze().detach().to('cpu').numpy())
            ax2.imshow(y[0,0,:,:].squeeze().detach().to('cpu').numpy())
            x_hat = d8Out[0,0,:,:].squeeze().detach().to('cpu').numpy()
            ax3.imshow(x_hat,vmin=x_hat.min(), vmax=x_hat.max())
            plt.show(block=False)
            plt.pause(1)
            plt.close()
        #        print(y[i,0,:,:].squeeze().detach().to('cpu').numpy())
        #        print(d8Out[i,0,:,:].squeeze().detach().to('cpu').numpy())
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # validation_step defines the validation loop.
        # it is independent of forward
        x, y, z = batch # reverberant speech, clean speech, RIR (RIR not used in this base UNet model)
        x = x[:,[0],:,:]
        y = y[:,[0],:,:]
        
        # each x = (256 x 256 x 1)
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

        loss   = nn.functional.mse_loss(d8Out, y)
        # Logging to TensorBoard by default
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        # test_step defines the test loop.
        # it is independent of forward
        x, y, z = batch # reverberant speech, clean speech, RIR (RIR not used in this base UNet model)
        x = x[:,[0],:,:]
        y = y[:,[0],:,:]
        
        # each x = (256 x 256 x 1)
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

        loss   = nn.functional.mse_loss(d8Out, y)
        # Logging to TensorBoard by default
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

############################################################
class Unet_Speech_RIR(pl.LightningModule):
    def __init__(self,learning_rate=1e-3):
        super().__init__()

        self.learning_rate = learning_rate

        # UNet model from "Speech Dereverberation Using Fully Convolutional Networks," Ernst et al., EUSIPCO 2018
        # with an additional decoding branch for the RIR
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
    
        self.rir1  = nn.Sequential(nn.ConvTranspose2d( 512, 512, 5, stride=2, padding=2, output_padding=1), nn.BatchNorm2d(512), nn.Dropout2d(p=0.5), nn.ReLU())
        self.rir2  = nn.Sequential(nn.ConvTranspose2d(1024, 512, 5, stride=2, padding=2, output_padding=1), nn.BatchNorm2d(512), nn.Dropout2d(p=0.5), nn.ReLU())
        self.rir3  = nn.Sequential(nn.ConvTranspose2d(1024, 512, 5, stride=2, padding=2, output_padding=1), nn.BatchNorm2d(512), nn.Dropout2d(p=0.5), nn.ReLU())
        self.rir4  = nn.Sequential(nn.ConvTranspose2d(1024, 512, 5, stride=2, padding=2, output_padding=1), nn.BatchNorm2d(512), nn.ReLU())
        self.rir5  = nn.Sequential(nn.ConvTranspose2d(1024, 256, 5, stride=2, padding=2, output_padding=1), nn.BatchNorm2d(256),  nn.ReLU())
        self.rir6  = nn.Sequential(nn.ConvTranspose2d( 512, 128, 5, stride=2, padding=2, output_padding=1), nn.BatchNorm2d(128),  nn.ReLU())
        self.rir7  = nn.Sequential(nn.ConvTranspose2d( 256,  64, 5, stride=2, padding=2, output_padding=1), nn.BatchNorm2d(64),  nn.ReLU())
        self.rir8  = nn.Sequential(nn.ConvTranspose2d( 128,   1, 5, stride=2, padding=2, output_padding=1), nn.Tanh())
        self.rir9  = nn.Flatten()
        self.rir10 = nn.Sequential(nn.Linear(65536, 4096), nn.Tanh()) # this layer causes a GPU memory overload

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y, z = batch # reverberant speech, clean speech, RIR (RIR not used in this base UNet model)
        x = x[:,[0],:,:]
        y = y[:,[0],:,:]

        # each x = (256 x 256 x 1)
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

        r1Out  = self.rir1(c8Out) # (2 x 2 x 1024)
        r2Out  = self.rir2(t.cat((r1Out, c7Out), dim=1)) # (4 x 4 x 1024)
        r3Out  = self.rir3(t.cat((r2Out, c6Out), dim=1)) # (8 x 8 x 1024)
        r4Out  = self.rir4(t.cat((r3Out, c5Out), dim=1)) # (16 x 16 x 1024)
        r5Out  = self.rir5(t.cat((r4Out, c4Out), dim=1)) # (32 x 32 x 512)
        r6Out  = self.rir6(t.cat((r5Out, c3Out), dim=1)) # (64 x 64 x 256)
        r7Out  = self.rir7(t.cat((r6Out, c2Out), dim=1)) # (128 x 128 x 128)
        r8Out  = self.rir8(t.cat((r7Out, c1Out), dim=1)) # (256 x 256 x 1)
        r9Out  = self.rir9(r8Out) # 65536 x 1
        r10Out = self.rir10(r9Out) # 4096 x 1 to match the first ~0.25s of the RIR size (2s @ 16 kHz)

        loss   = nn.functional.mse_loss(d8Out, y) + nn.functional.mse_loss(r10Out, z[:, 0:4096])

        #if batch_idx==0 or batch_idx==1000:
        import matplotlib.pyplot as plt
        if batch_idx == 9:
        #    for i in range(2):
        #        plt.imshow(x[i,0,:,:].squeeze().to('cpu').numpy())
        #        plt.show()
        #        plt.imshow(y[i,0,:,:].squeeze().to('cpu').numpy())
        #        plt.show()
            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
            fig.set_figwidth(3*fig.get_figwidth())
            ax1.imshow(x[0,0,:,:].squeeze().detach().to('cpu').numpy())
            ax2.imshow(y[0,0,:,:].squeeze().detach().to('cpu').numpy())
            x_hat = d8Out[0,0,:,:].squeeze().detach().to('cpu').numpy()
            ax3.imshow(x_hat,vmin=x_hat.min(), vmax=x_hat.max())
            ax4.plot(z[0,:].squeeze().detach().to('cpu').numpy())
            ax5.plot(r9Out[0, 0:32000].squeeze().detach().to('cpu').numpy())

            plt.savefig('DAREoutput_epoch_' + str(self.current_epoch) + '.pdf', bbox_inches='tight')
            plt.show(block=False)
            plt.pause(0.1)
            plt.close()
        #        print(y[i,0,:,:].squeeze().detach().to('cpu').numpy())
        #        print(d8Out[i,0,:,:].squeeze().detach().to('cpu').numpy())
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # validation_step defines the validation loop.
        # it is independent of forward
        x, y, z = batch # reverberant speech, clean speech, RIR (RIR not used in this base UNet model)
        x = x[:,[0],:,:]
        y = y[:,[0],:,:]
        
        # each x = (256 x 256 x 1)
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

        r1Out  = self.rir1(c8Out) # (2 x 2 x 1024)
        r2Out  = self.rir2(t.cat((r1Out, c7Out), dim=1)) # (4 x 4 x 1024)
        r3Out  = self.rir3(t.cat((r2Out, c6Out), dim=1)) # (8 x 8 x 1024)
        r4Out  = self.rir4(t.cat((r3Out, c5Out), dim=1)) # (16 x 16 x 1024)
        r5Out  = self.rir5(t.cat((r4Out, c4Out), dim=1)) # (32 x 32 x 512)
        r6Out  = self.rir6(t.cat((r5Out, c3Out), dim=1)) # (64 x 64 x 256)
        r7Out  = self.rir7(t.cat((r6Out, c2Out), dim=1)) # (128 x 128 x 128)
        r8Out  = self.rir8(t.cat((r7Out, c1Out), dim=1)) # (256 x 256 x 1)
        r9Out  = self.rir9(r8Out) # 65536 x 1
        r10Out = self.rir10(r9Out) # 4096 x 1 

        #loss   = nn.functional.mse_loss(d8Out, y) + nn.functional.mse_loss(r10Out, z)
        loss   = nn.functional.mse_loss(d8Out, y) + nn.functional.mse_loss(r10Out, z[:, 0:4096])

        #if batch_idx==0 or batch_idx==1000:
        # Logging to TensorBoard by default
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        # test_step defines the test loop.
        # it is independent of forward
        x, y, z = batch # reverberant speech, clean speech, RIR (RIR not used in this base UNet model)
        x = x[:,[0],:,:]
        y = y[:,[0],:,:]
        
        # each x = (256 x 256 x 1)
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

        r1Out  = self.rir1(c8Out) # (2 x 2 x 1024)
        r2Out  = self.rir2(t.cat((r1Out, c7Out), dim=1)) # (4 x 4 x 1024)
        r3Out  = self.rir3(t.cat((r2Out, c6Out), dim=1)) # (8 x 8 x 1024)
        r4Out  = self.rir4(t.cat((r3Out, c5Out), dim=1)) # (16 x 16 x 1024)
        r5Out  = self.rir5(t.cat((r4Out, c4Out), dim=1)) # (32 x 32 x 512)
        r6Out  = self.rir6(t.cat((r5Out, c3Out), dim=1)) # (64 x 64 x 256)
        r7Out  = self.rir7(t.cat((r6Out, c2Out), dim=1)) # (128 x 128 x 128)
        r8Out  = self.rir8(t.cat((r7Out, c1Out), dim=1)) # (256 x 256 x 1)
        r9Out  = self.rir9(r8Out) # 65536 x 1
        r10Out = self.rir10(r9Out) # 4096 x 1 

        #loss   = nn.functional.mse_loss(d8Out, y)  + nn.functional.mse_loss(r10Out, z)
        loss   = nn.functional.mse_loss(d8Out, y) + nn.functional.mse_loss(r10Out, z[:, 0:4096])

        #if batch_idx==0 or batch_idx==1000:
        # Logging to TensorBoard by default
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

############################################################
class Unet_Speech_RIR_2(pl.LightningModule):
    def __init__(self,learning_rate=1e-3):
        super().__init__()

        self.learning_rate = learning_rate

        # UNet model from "Speech Dereverberation Using Fully Convolutional Networks," Ernst et al., EUSIPCO 2018
        # with an additional decoding branch for the RIR
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
    
        # Not sure about this
        #self.rir1 = nn.Sequential(nn.ConvTranspose2d(512, 1024, 1))
        #self.rir2 = nn.Sequential(nn.ConvTranspose2d(1024, 2048, 1))
        #self.rir3 = nn.Sequential(nn.ConvTranspose2d(2048, 4096, 1))
        #self.rir4 = nn.Sequential(nn.ConvTranspose2d(4096, 8192, 1))
        #self.rir5 = nn.Sequential(nn.ConvTranspose2d(8192, 16384, 1))
        #self.rir6 = nn.Sequential(nn.ConvTranspose2d(16384, 32000, 1))
        #self.rir7 = nn.Flatten()

        # This doesn't work, the shapes are wrong 
        self.rir1 = nn.Flatten()
        self.rir2 = nn.Sequential(nn.ConvTranspose1d(1, 1, 5, stride=2, padding=2, output_padding=1), nn.Tanh())
        self.rir3 = nn.Sequential(nn.ConvTranspose1d(1, 1, 5, stride=2, padding=2, output_padding=1), nn.Tanh())
        self.rir4 = nn.Sequential(nn.ConvTranspose1d(1, 1, 5, stride=2, padding=2, output_padding=1), nn.Tanh())
        self.rir5 = nn.Sequential(nn.ConvTranspose1d(1, 1, 5, stride=2, padding=2, output_padding=1), nn.Tanh())
        self.rir6 = nn.Sequential(nn.ConvTranspose1d(1, 1, 5, stride=2, padding=2, output_padding=1), nn.Tanh())
        self.rir7 = nn.Sequential(nn.ConvTranspose1d(1, 1, 5, stride=2, padding=2, output_padding=1), nn.Tanh())

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y, z = batch # reverberant speech, clean speech, RIR (RIR not used in this base UNet model)
        x = x[:,[0],:,:]
        y = y[:,[0],:,:]

        # each x = (256 x 256 x 1)
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

        r1Out  = self.rir1(c8Out) # (1 x 1 x  1024)
        r2Out  = self.rir2(r1Out) # (1 x 1 x  2048)
        r3Out  = self.rir3(r2Out) # (1 x 1 x  4096)
        r4Out  = self.rir4(r3Out) # (1 x 1 x  8192)
        r5Out  = self.rir5(r4Out) # (1 x 1 x 16384)
        r6Out  = self.rir6(r5Out) # (1 x 1 x 32000)
        r7Out  = self.rir7(r6Out) # flattened
 
        loss   = nn.functional.mse_loss(d8Out, y) + nn.functional.mse_loss(r7Out, z)

        #if batch_idx==0 or batch_idx==1000:
        import matplotlib.pyplot as plt
        if batch_idx == 9:
        #    for i in range(2):
        #        plt.imshow(x[i,0,:,:].squeeze().to('cpu').numpy())
        #        plt.show()
        #        plt.imshow(y[i,0,:,:].squeeze().to('cpu').numpy())
        #        plt.show()
            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
            fig.set_figwidth(3*fig.get_figwidth())
            ax1.imshow(x[0,0,:,:].squeeze().detach().to('cpu').numpy())
            ax2.imshow(y[0,0,:,:].squeeze().detach().to('cpu').numpy())
            x_hat = d8Out[0,0,:,:].squeeze().detach().to('cpu').numpy()
            ax3.imshow(x_hat,vmin=x_hat.min(), vmax=x_hat.max())
            ax4.plot(z[0,:].squeeze().detach().to('cpu').numpy())
            ax5.plot(r9Out[0, 0:32000].squeeze().detach().to('cpu').numpy())

            plt.savefig('DAREoutput_epoch_' + str(self.current_epoch) + '.pdf', bbox_inches='tight')
            plt.show(block=False)
            plt.pause(0.1)
            plt.close()
        #        print(y[i,0,:,:].squeeze().detach().to('cpu').numpy())
        #        print(d8Out[i,0,:,:].squeeze().detach().to('cpu').numpy())
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # validation_step defines the validation loop.
        # it is independent of forward
        x, y, z = batch # reverberant speech, clean speech, RIR (RIR not used in this base UNet model)
        x = x[:,[0],:,:]
        y = y[:,[0],:,:]
        
        # each x = (256 x 256 x 1)
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

        r1Out  = self.rir1(c8Out) # (1 x 1 x  1024)
        r2Out  = self.rir2(r1Out) # (1 x 1 x  2048)
        r3Out  = self.rir3(r2Out) # (1 x 1 x  4096)
        r4Out  = self.rir4(r3Out) # (1 x 1 x  8192)
        r5Out  = self.rir5(r4Out) # (1 x 1 x 16384)
        r6Out  = self.rir6(r5Out) # (1 x 1 x 32000)
        r7Out  = self.rir7(r6Out) # flattened

        loss   = nn.functional.mse_loss(d8Out, y) + nn.functional.mse_loss(r7Out, z)

        #if batch_idx==0 or batch_idx==1000:
        # Logging to TensorBoard by default
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        # test_step defines the test loop.
        # it is independent of forward
        x, y, z = batch # reverberant speech, clean speech, RIR (RIR not used in this base UNet model)
        x = x[:,[0],:,:]
        y = y[:,[0],:,:]
        
        # each x = (256 x 256 x 1)
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

        r1Out  = self.rir1(c8Out) # (1 x 1 x  1024)
        r2Out  = self.rir2(r1Out) # (1 x 1 x  2048)
        r3Out  = self.rir3(r2Out) # (1 x 1 x  4096)
        r4Out  = self.rir4(r3Out) # (1 x 1 x  8192)
        r5Out  = self.rir5(r4Out) # (1 x 1 x 16384)
        r6Out  = self.rir6(r5Out) # (1 x 1 x 32000)
        r7Out  = self.rir7(r6Out) # flattened 

        loss   = nn.functional.mse_loss(d8Out, y)  + nn.functional.mse_loss(r7Out, z)

        #if batch_idx==0 or batch_idx==1000:
        # Logging to TensorBoard by default
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

