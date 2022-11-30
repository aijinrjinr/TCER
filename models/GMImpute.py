import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from models.deconv import FastDeconv
from models.DCNv2.dcn_v2 import DCN
from models.cbam import CBAM

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y






class DehazeBlock_dcn(nn.Module):
    def __init__(self, conv, dim, kernel_size, ):
        super(DehazeBlock_dcn, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        # self.conv1 = DCN(dim, dim, kernel_size=kernel_size, stride=1, padding=1)  #
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.conv3 = DCN(dim, dim, kernel_size=kernel_size, stride=1, padding=1)#DCNBlock(dim, dim)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.conv3(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res





class GMImpute(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64):
        super(GMImpute, self).__init__()

        ###### downsample
        self.down1 = nn.Sequential(nn.ReflectionPad2d(3),
                                   nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                                   nn.ReLU(True))
        self.down2 = nn.Sequential(nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        self.down3 = nn.Sequential(nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))

        ###### DFA blocks
        self.block_0 = DehazeBlock_dcn(default_conv, ngf * 4, 3)#
        self.block_1 = DehazeBlock_dcn(default_conv, ngf * 4, 3)
        self.block_2 = DehazeBlock_dcn(default_conv, ngf * 4, 3)


        ###### upsample
        self.up1 = nn.Sequential(nn.ConvTranspose2d(ngf*8, ngf*2, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(ngf*4, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),#)
                                 nn.ReLU(True))
        self.up3 = nn.Sequential(nn.ReflectionPad2d(3),
                                 nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                 # nn.Sigmoid())
                                 nn.Tanh())



        self.deconv = FastDeconv(1, 1, kernel_size=3, stride=1, padding=1)


        self.fusion1 = CBAM(256 * 2)#FeaAtt(default_conv, 256 * 2, 3)#CBAM(256 * 2)
        self.fusion2 = CBAM(128 * 2)#FeaAtt(default_conv, 128 * 2, 3)#CBAM(128 * 2)



    def forward(self, input):

        x_deconv = self.deconv(input) # preprocess

        x_down1 = self.down1(x_deconv) # [bs, 64, 256, 256]
        x_down2 = self.down2(x_down1) # [bs, 128, 128, 128]
        x_down3 = self.down3(x_down2) # [bs, 256, 64, 64]



        x1 = self.block_0(x_down3)

        x2 = self.block_1(x1)

        x3 = self.block_2(x2)


        x_out_mix = torch.cat([x3, x_down3], dim=1)#self.mix1(x_down3, x_dcn2)
        x_out_mix = self.fusion1(x_out_mix)
        x_up1 = self.up1(x_out_mix) # [bs, 128, 128, 128]

        x_up1_mix = torch.cat([x_up1, x_down2], dim=1)
        x_up1_mix = self.fusion2(x_up1_mix)
        x_up2 = self.up2(x_up1_mix) # [bs, 64, 256, 256]

        out = self.up3(x_up2) # [bs,  3, 256, 256]

        return out









