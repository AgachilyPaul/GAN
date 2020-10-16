import torch
import torch.nn as nn
import torch.nn.functional as F

class conv(torch.nn.Sequential):
    def __init__(self, in_dim, out_dim, k_size, s, p):
        super(conv, self).__init__()
        self.conv_layer = nn.Conv2d(in_dim, out_dim, k_size, s, p,)
        #self.bn_layer = nn.BatchNorm2d(out_dim)
        self.lrelu_layer = nn.LeakyReLU(0.2)
        

class deconv(torch.nn.Sequential):
    def __init__(self, in_dim, out_dim, k_size, s, p, o):
        super(deconv, self).__init__()
        self.deconv_layer = nn.ConvTranspose2d(in_dim, out_dim, k_size, s, p, o)
        self.relu_layer = nn.LeakyReLU(0.2)


class Discriminator(torch.nn.Module):
    def __init__(self, cfg=None, logger=None):
        super(Discriminator, self).__init__()
        self.logger = logger
        self.basic_dim = 64
        self.h1 = nn.Sequential(nn.Conv2d(3, 32, 5, 2, 2),
                               nn.LeakyReLU(0.2),)
        self.h2 = conv(32, 64, 5, 1, 2)
        self.h3 = conv(64, 64, 5, 2, 2)
        self.h4 = conv(64, 128, 5, 1, 2)
        self.h5 = conv(128, 128, 5, 4, 2)
        self.h6 = conv(128, 256, 5, 1, 2)
        self.h7 = conv(256, 256, 5, 4, 2)
        self.h8 = conv(256, 512, 5, 1, 2)
        self.h9 = conv(512, 512, 5, 4, 2)
        self.flat = nn.Flatten()
        self.h10 = nn.Linear(512, 1)
        self.sig = nn.Sigmoid()
        self._init()

    def forward(self, inputs):
        x = self.h1(inputs)
        x = self.h2(x)
        x = self.h3(x)
        x = self.h4(x)
        x = self.h5(x)
        x = self.h6(x)
        x = self.h7(x)
        x = self.h8(x)
        x = self.h9(x)
        x = self.flat(x)
        x = self.h10(x)
        x_sig = self.sig(x)       
        return x_sig

    def _init(self):
        for name, sub_module in self.named_modules():
            if isinstance(sub_module, nn.Conv2d) or isinstance(sub_module, nn.ConvTranspose2d):
                nn.init.normal_(sub_module.weight, std=0.02)
                if sub_module.bias is not None:
                    nn.init.constant_(sub_module.bias, 0.0)
                self.logger.info('init {}.weight as normal(0, 0.002)'.format(name))
                self.logger.info('init {}.bias as 0'.format(name))

