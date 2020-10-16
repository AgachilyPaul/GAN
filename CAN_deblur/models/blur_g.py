
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, k_size, s, p):
        super(ResBlock, self).__init__()
        if in_dim!=out_dim or s!=1:
            self.downsample = nn.Conv2d(in_dim, out_dim, k_size, s, p)
        else:
            self.downsample = None
        self.conv_layer_0 = nn.Conv2d(in_dim, out_dim, k_size, s, p,
                                    bias=True)
        self.relu_layer_0 = nn.ReLU()
        self.conv_layer_1 = nn.Conv2d(out_dim, out_dim, k_size, 1, p,
                                    bias=True)

    def forward(self, input_):
        inputcheck = input_
        x = self.conv_layer_0(input_)
        x = self.relu_layer_0(x)
        x = self.conv_layer_1(x)
        if self.downsample is not None:
            inputcheck = self.downsample(input_)
        x += inputcheck
        return x    
      

class deconv(torch.nn.Sequential):
    def __init__(self, in_dim, out_dim, k_size, s, p, o):
        super(deconv, self).__init__()
        self.relu_layer = nn.ReLU()
        self.deconv_layer = nn.ConvTranspose2d(in_dim, out_dim, k_size, s, p, o)
        self.bn_layer = nn.BatchNorm2d(out_dim)
        

class Generator(torch.nn.Module):
    def __init__(self, cfg=None, logger=None):
        super(Generator, self).__init__()
        self.logger = logger
        self.b3_l3_in = nn.Conv2d(3, 64, 5, 1, 2)
        self.b3_l3_res = self._make_layer(5, 2, [(64,64,1),(64,64,1),(64,64,1),
            (64,64,1),(64,64,1),(64,64,1),(64,64,1),(64,64,1),(64,64,1),
            (64,64,1),(64,64,1),(64,64,1),(64,64,1),(64,64,1),(64,64,1),])
        self.b3_l3_out = nn.Conv2d(64, 3, 5, 1, 2)

        self.b2_l2_in = nn.Conv2d(6, 64, 5, 1, 2)
        self.b2_l2_res = self._make_layer(5, 2, [(64,64,1),(64,64,1),(64,64,1),
            (64,64,1),(64,64,1),(64,64,1),(64,64,1),(64,64,1),(64,64,1),
            (64,64,1),(64,64,1),(64,64,1),(64,64,1),(64,64,1),(64,64,1),])
        self.b2_l2_out = nn.Conv2d(64, 3, 5, 1, 2)

        self.b1_l1_in = nn.Conv2d(6, 64, 5, 1, 2)
        self.b1_l1_res = self._make_layer(5, 2, [(64,64,1),(64,64,1),(64,64,1),
            (64,64,1),(64,64,1),(64,64,1),(64,64,1),(64,64,1),(64,64,1),
            (64,64,1),(64,64,1),(64,64,1),(64,64,1),(64,64,1),(64,64,1),])
        self.b1_l1_out = nn.Conv2d(64, 3, 5, 1, 2)
        self._init()
    
    def _make_layer(self, k_size, padding, block_list):
        layers = []
        for block_cfg in block_list:
            layers.append(ResBlock(block_cfg[0], block_cfg[1], 
                k_size, block_cfg[2], padding))
        return nn.Sequential(*layers)
          
    def forward(self, in_256, in_128, in_64):
        #self.logger.debug(in_64.shape)
        x_64 = self.b3_l3_in(in_64)
        x_64 = self.b3_l3_res(x_64)
        x_64 = self.b3_l3_out(x_64)

        x_128_up = F.upsample(x_64, scale_factor=2)
        x_128 = torch.cat([x_128_up,in_128],1)
        x_128 = self.b2_l2_in(x_128)
        x_128 = self.b2_l2_res(x_128)
        x_128 = self.b2_l2_out(x_128)

        x_256_up = F.upsample(x_128, scale_factor=2)
        #print(x_256_up.shape)
        x_256 = torch.cat([x_256_up,in_256],1)
        x_256 = self.b1_l1_in(x_256)
        x_256 = self.b1_l1_res(x_256)
        x_256 = self.b1_l1_out(x_256)

        return [F.tanh(x_256), F.tanh(x_128), F.tanh(x_64)]

    def _init(self):
        for name, sub_module in self.named_modules():
            if isinstance(sub_module, nn.Conv2d) or isinstance(sub_module, nn.ConvTranspose2d):
                nn.init.normal_(sub_module.weight, std=0.02)
                if sub_module.bias is not None:
                    nn.init.constant_(sub_module.bias, 0.0)
                self.logger.info('init {}.weight as normal(0, 0.002)'.format(name))
                self.logger.info('init {}.bias as 0'.format(name))