import torch
#import torch.nn as nn
from torch import optim, nn
import pytorch_lightning as pl

from WaveUnet.crop import centre_crop
from WaveUnet.resample import Resample1d
from WaveUnet.conv import ConvLayer

import auraloss # for MR-STFT loss 

class UpsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, stride, depth, conv_type, res):
        super(UpsamplingBlock, self).__init__()
        assert(stride > 1)

        # CONV 1 for UPSAMPLING
        if res == "fixed":
            self.upconv = Resample1d(n_inputs, 15, stride, transpose=True)
        else:
            self.upconv = ConvLayer(n_inputs, n_inputs, kernel_size, stride, conv_type, transpose=True)

        self.pre_shortcut_convs = nn.ModuleList([ConvLayer(n_inputs, n_outputs, kernel_size, 1, conv_type)] +
                                                [ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type) for _ in range(depth - 1)])

        # CONVS to combine high- with low-level information (from shortcut)
        self.post_shortcut_convs = nn.ModuleList([ConvLayer(n_outputs + n_shortcut, n_outputs, kernel_size, 1, conv_type)] +
                                                 [ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type) for _ in range(depth - 1)])

    def forward(self, x, shortcut):
        # UPSAMPLE HIGH-LEVEL FEATURES
        upsampled = self.upconv(x)

        for conv in self.pre_shortcut_convs:
            upsampled = conv(upsampled)

        # Prepare shortcut connection
        combined = centre_crop(shortcut, upsampled)

        # Combine high- and low-level features
        for conv in self.post_shortcut_convs:
            combined = conv(torch.cat([combined, centre_crop(upsampled, combined)], dim=1))
        return combined

    def get_output_size(self, input_size):
        curr_size = self.upconv.get_output_size(input_size)

        # Upsampling convs
        for conv in self.pre_shortcut_convs:
            curr_size = conv.get_output_size(curr_size)

        # Combine convolutions
        for conv in self.post_shortcut_convs:
            curr_size = conv.get_output_size(curr_size)

        return curr_size

class DownsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, stride, depth, conv_type, res):
        super(DownsamplingBlock, self).__init__()
        assert(stride > 1)

        self.kernel_size = kernel_size
        self.stride = stride

        # CONV 1
        self.pre_shortcut_convs = nn.ModuleList([ConvLayer(n_inputs, n_shortcut, kernel_size, 1, conv_type)] +
                                                [ConvLayer(n_shortcut, n_shortcut, kernel_size, 1, conv_type) for _ in range(depth - 1)])

        self.post_shortcut_convs = nn.ModuleList([ConvLayer(n_shortcut, n_outputs, kernel_size, 1, conv_type)] +
                                                 [ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type) for _ in
                                                  range(depth - 1)])

        # CONV 2 with decimation
        if res == "fixed":
            self.downconv = Resample1d(n_outputs, 15, stride) # Resampling with fixed-size sinc lowpass filter
        else:
            self.downconv = ConvLayer(n_outputs, n_outputs, kernel_size, stride, conv_type)

    def forward(self, x):
        # PREPARING SHORTCUT FEATURES
        shortcut = x
        for conv in self.pre_shortcut_convs:
            shortcut = conv(shortcut)

        # PREPARING FOR DOWNSAMPLING
        out = shortcut
        for conv in self.post_shortcut_convs:
            out = conv(out)

        # DOWNSAMPLING
        out = self.downconv(out)

        return out, shortcut

    def get_input_size(self, output_size):
        curr_size = self.downconv.get_input_size(output_size)

        for conv in reversed(self.post_shortcut_convs):
            curr_size = conv.get_input_size(curr_size)

        for conv in reversed(self.pre_shortcut_convs):
            curr_size = conv.get_input_size(curr_size)
        return curr_size

class Waveunet(pl.LightningModule):
    def __init__(self, num_inputs, num_channels, num_outputs, instruments, kernel_size_down, kernel_size_up, target_output_size, conv_type, res, separate=False, depth=1, strides=2, learning_rate=0.0001):
        super(Waveunet, self).__init__()

        self.name = "Waveunet"
        self.num_levels = len(num_channels)
        self.strides = strides
        self.kernel_size_down = kernel_size_down
        self.kernel_size_up = kernel_size_up
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.depth = depth
        self.instruments = instruments
        self.separate = separate
        self.learning_rate = learning_rate

        # Only odd filter kernels allowed
        assert(kernel_size_down % 2 == 1)
        assert(kernel_size_up   % 2 == 1)

        self.waveunets = nn.ModuleDict()

        model_list = instruments if separate else ["ALL"]
        # Create a model for each source if we separate sources separately, otherwise only one (model_list=["ALL"])
        for instrument in model_list:
            module = nn.Module()

            module.downsampling_blocks = nn.ModuleList()
            module.upsampling_blocks = nn.ModuleList()

            for i in range(self.num_levels - 1):
                in_ch = num_inputs if i == 0 else num_channels[i]

                module.downsampling_blocks.append(
                    DownsamplingBlock(in_ch, num_channels[i], num_channels[i+1], kernel_size_down, strides, depth, conv_type, res))

            for i in range(0, self.num_levels - 1):
                module.upsampling_blocks.append(
                    UpsamplingBlock(num_channels[-1-i], num_channels[-2-i], num_channels[-2-i], kernel_size_up, strides, depth, conv_type, res))

            module.bottlenecks = nn.ModuleList(
                [ConvLayer(num_channels[-1], num_channels[-1], kernel_size_down, 1, conv_type) for _ in range(depth)])

            # Output conv
            outputs = num_outputs if separate else num_outputs * len(instruments)
            module.output_conv = nn.Conv1d(num_channels[0], outputs, 1)

            self.waveunets[instrument] = module

        self.set_output_size(target_output_size)

    def set_output_size(self, target_output_size):
        self.target_output_size = target_output_size

        #print("<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>")
        #print("target_output_size = " + str(target_output_size))
        #print("<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>")

        self.input_size, self.output_size = self.check_padding(target_output_size)
        print("Using valid convolutions with " + str(self.input_size) + " inputs and " + str(self.output_size) + " outputs")

        assert((self.input_size - self.output_size) % 2 == 0)
        self.shapes = {"output_start_frame" : (self.input_size - self.output_size) // 2,
                       "output_end_frame" : (self.input_size - self.output_size) // 2 + self.output_size,
                       "output_frames" : self.output_size,
                       "input_frames" : self.input_size}

    def check_padding(self, target_output_size):
        # Ensure number of outputs covers a whole number of cycles so
        # each output in the cycle is weighted equally during training
        bottleneck = 1

        while True:
            out = self.check_padding_for_bottleneck(bottleneck, target_output_size)
            if out is not False:
                return out
            bottleneck += 1

    def check_padding_for_bottleneck(self, bottleneck, target_output_size):
        module = self.waveunets[[k for k in self.waveunets.keys()][0]]
        try:
            curr_size = bottleneck
            for idx, block in enumerate(module.upsampling_blocks):
                curr_size = block.get_output_size(curr_size)
            output_size = curr_size

            # Bottleneck-Conv
            curr_size = bottleneck
            for block in reversed(module.bottlenecks):
                curr_size = block.get_input_size(curr_size)
            for idx, block in enumerate(reversed(module.downsampling_blocks)):
                curr_size = block.get_input_size(curr_size)

            assert(output_size >= target_output_size)
            return curr_size, output_size
        except AssertionError as e:
            return False

    def forward_module(self, x, module):
        '''
        A forward pass through a single Wave-U-Net (multiple Wave-U-Nets might be used, one for each source)
        :param x: Input mix
        :param module: Network module to be used for prediction
        :return: Source estimates
        '''
        shortcuts = []
        out = x

        # DOWNSAMPLING BLOCKS
        for block in module.downsampling_blocks:
            out, short = block(out)
            shortcuts.append(short)

        # BOTTLENECK CONVOLUTION
        for conv in module.bottlenecks:
            out = conv(out)

        # UPSAMPLING BLOCKS
        for idx, block in enumerate(module.upsampling_blocks):
            out = block(out, shortcuts[-1 - idx])

        # OUTPUT CONV
        out = module.output_conv(out)
        if not self.training:  # At test time clip predictions to valid amplitude range
            out = out.clamp(min=-1.0, max=1.0)
        return out

    def forward(self, x, inst=None):
        curr_input_size = x.shape[-1]
        #print("********************")
        #print("input shape = " + str(x.shape))
        #print("curr_input_size = " + str(curr_input_size))
        #print("self.input_size = " + str(self.input_size))
        #print("self.output_size = " + str(self.output_size))
        #print("********************")
        assert(curr_input_size == self.input_size) # User promises to feed the proper input himself, to get the pre-calculated (NOT the originally desired) output size

        if self.separate:
            return {inst : self.forward_module(x, self.waveunets[inst])}
        else:
            assert(len(self.waveunets) == 1)
            out = self.forward_module(x, self.waveunets["ALL"])

            out_dict = {}
            for idx, inst in enumerate(self.instruments):
                out_dict[inst] = out[:, idx * self.num_outputs:(idx + 1) * self.num_outputs]
            return out_dict

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward (but uses it)
        x, y, z = batch # reverberant speech, clean speech, RIR # should be all time domain


        # convert from (batch_size, num_samples) to (batch_size, 1, num_samples)
        x = x[:, None, :].float()
        y = y[:, None, :].float()
        z = z[:, None, :].float()

        #print('x.shape = ' + str(x.shape))
        #print('y.shape = ' + str(y.shape))
        #print('z.shape = ' + str(z.shape))        
        
        #y_hat, z_hat = self.predict(x)
        out  = self.forward(x)
        #print("out[speech].shape = " + str(out["speech"].shape))
        #print("out[rir].shape = " + str(out["rir"].shape))
        #loss = nn.functional.mse_loss(out["speech"], y) + nn.functional.mse_loss(out["rir"], z)
        
        speechLoss = nn.functional.mse_loss(out["speech"], centre_crop(y, out["speech"]))
        rirLoss    = nn.functional.mse_loss(out["rir"], z)
        
        #loss = nn.functional.mse_loss(out["speech"], centre_crop(y, out["speech"])) + nn.functional.mse_loss(out["rir"], z)
        #loss = speechLoss + 5.0*rirLoss # 5x RIR seems to put them in better balance but worked poorly
        loss = speechLoss # Can I get better clean speech by ignoring the RIR?
        
        #MR-STFT loss
        fft_sizes   = [16, 128, 512, 2048]
        hop_sizes   = [ 8,  64, 256, 1024]
        win_lengths = [16, 128, 512, 2048]
        
        #mrstftLoss = auraloss.freq.MultiResolutionSTFTLoss(fft_sizes=fft_sizes, hop_sizes=hop_sizes, win_lengths=win_lengths)
        # Not sure yet how to balance the 2 terms
        #loss = nn.functional.mse_loss(out["speech"], centre_crop(y, out["speech"])) + mrstftLoss(out["rir"], z)

        #print("Speech loss = " + str(speechLoss))
        #print("RIR loss    = " + str(rirLoss))

        self.log("loss", {'train': loss })
        self.log("train_loss", loss )
        self.log("train_speechLoss", speechLoss )
        self.log("train_rirLoss", rirLoss )

        return loss

    def validation_step(self, batch, batch_idx):
        # validation_step defines the train loop.
        # it is independent of forward (but uses it)
        x, y, z = batch # reverberant speech, clean speech, RIR # should be all time domain

        # convert from (batch_size, num_samples) to (batch_size, 1, num_samples)
        x = x[:, None, :].float()
        y = y[:, None, :].float()
        z = z[:, None, :].float()

        out  = self.forward(x)
        speechLoss = nn.functional.mse_loss(out["speech"], centre_crop(y, out["speech"]))
        rirLoss    = nn.functional.mse_loss(out["rir"], z)
        #loss = speechLoss + 5.0*rirLoss
        loss = speechLoss
        
        self.log("loss", {'val': loss })
        self.log("val_loss", loss )
        self.log("val_speechLoss", speechLoss )
        self.log("val_rirLoss", rirLoss )

        return loss

    def test_step(self, batch, batch_idx):
        # test_step for trainer.test()
        # it is independent of forward (but uses it)
        x, y, z = batch # reverberant speech, clean speech, RIR # should be all time domain

        # convert from (batch_size, num_samples) to (batch_size, 1, num_samples)
        x = x[:, None, :].float()
        y = y[:, None, :].float()
        z = z[:, None, :].float()

        out  = self.forward(x)
        speechLoss = nn.functional.mse_loss(out["speech"], centre_crop(y, out["speech"]))
        rirLoss    = nn.functional.mse_loss(out["rir"], z)
        #loss = speechLoss + 5.0*rirLoss
        loss = speechLoss
        
        # Try nn.functional.l1_loss() for the RIR
        # Try a multiresolution FFT loss per Chrisitan's auraloss library

        self.log("loss", {'test': loss })
        self.log("test_loss", loss )
        self.log("test_speechLoss", speechLoss )
        self.log("test_rirLoss", rirLoss )

        return loss


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer