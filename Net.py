import torch
import torch.nn as nn

#卷积模块
class Conv(nn.Module):
    def __init__(self,in_c,out_c,k,s,p):
        super(Conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c,out_c,k,s,p),
            nn.ReLU()
        )
    def forward(self,x):
        return self.conv(x)

#残差模块
class ResidualBlock(torch.nn.Module):
    def __init__(self):
        super(ResidualBlock,self).__init__()
        self.conv = nn.Sequential(
            Conv(64,128,3,1,1),
            Conv(128,256,3,1,1),
            Conv(256,256,3,1,1),
            Conv(256,128,3,1,1),
            Conv(128,64,3,1,1),)

    def forward(self, x):
        y = self.conv(x)
        return x+y

# 模型
class model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.illumination_net = nn.Sequential(
            nn.Conv2d(1,64,3,1,1),
            ResidualBlock(),
            nn.Conv2d(64,1,3,1,1),
        )
        self.reflectance_net = nn.Sequential(
            nn.Conv2d(1,64,3,1,1),
            ResidualBlock(),
            nn.Conv2d(64,1,3,1,1),
        )
    def forward(self,x):
        I = torch.sigmoid(self.illumination_net(x.to(torch.float32)))
        R = torch.sigmoid(self.reflectance_net(x.to(torch.float32)))
        return I,R

#残差模块
class ResB(nn.Module):
    def __init__(self,in_c,out_c):
        super(ResB,self).__init__()
        self.Res = nn.Sequential(
            nn.Conv2d(in_c,out_c,1,1,0),
            nn.LeakyReLU(),
            nn.Conv2d(out_c,out_c,3,1,1),
            nn.LeakyReLU(),
            nn.Conv2d(out_c,out_c,1,1,0),
        )
        self.Conv = nn.Conv2d(in_c,out_c,1,1,0)
        self.activate = nn.LeakyReLU()
    def forward(self, x):
        y = self.Res(x) + self.Conv(x)
        y = self.activate(y)
        return y

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Encoder(nn.Module):
    def __init__(self) -> None:
        super(Encoder,self).__init__()
        self.Encoder_R = nn.Sequential(
            nn.Conv2d(1,64,5,1,2),
            nn.LeakyReLU(),
            ResB(64,64),
            ResB(64,64),
            ResB(64,64),
        )
        self.Encoder_I = nn.Sequential(
            nn.Conv2d(1,64,5,1,2),
            nn.LeakyReLU(),
            ResB(64,64),
            ResB(64,64),
            ResB(64,64),
        )
    def forward(self,R,I):
        return self.Encoder_R(R),self.Encoder_I(I)

class Decoder(nn.Module):
    def __init__(self) -> None:
        super(Decoder,self).__init__()
        self.Decoder = nn.Sequential(
            ResB(64,64),
            ResB(64,64),
            ResB(64,64),
            ResB(64,1),
        )
    def forward(self,input):
        return self.Decoder(input)
    
class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


class ABF_block(nn.Module):
    def __init__(self) -> None:
        super(ABF_block,self).__init__()
        self.CBAM = cbam_block(64)
        self.Conv = nn.Conv2d(64,64,3,1,1)
        self.Resb = ResB(128,64)
    
    def forward(self,fr1,fi1,fr2,fi2):
        f1 = self.CBAM(self.Conv(fr1*fi1))
        f2 = self.CBAM(self.Conv(fr2*fi2))
        f  = self.Resb(torch.cat([f1,f2],1))
        return f
    