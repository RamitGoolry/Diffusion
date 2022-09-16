import torch
import torch.nn as nn
import torch.nn.functional as F

from icecream import ic

class GroupNorm(nn.Module):
    '''
    Group Normalization Layer
    '''
    def __init__(self, channels):
        super(GroupNorm, self).__init__()
        
        self.gn = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)
    
    def forward(self, x):
        return self.gn(x)


class Swish(nn.Module):
    '''
    Swish Activation Layer
    -----------------------------
    Similar to ReLU, but is actually continuous so much more differentiable.
    Leads to much better performance across a wide range of tasks.
    '''
    def forward(self, x):
        return x * torch.sigmoid(x)

class ResidualBlock(nn.Module):
    '''
    Residual Block
    -----------------------------

    Main Building Block of a ResNet. Uses Skip connections to pass information along.
    '''
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block = nn.Sequential(
            GroupNorm(in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            GroupNorm(out_channels),
            Swish(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )

        if in_channels != out_channels:
            self.channel_up = nn.Conv2d(in_channels, out_channels, 1, 1, 0) # Channel Translation Layer

    def forward(self, x):
        if self.in_channels != self.out_channels:
            return self.channel_up(x) + self.block(x)
        else:
            return x + self.block(x)

class UpSampleBlock(nn.Module):
    '''
    Upsamples the given input to 2x its size and then runs a convolution
    '''
    def __init__(self, channels):
        super(UpSampleBlock, self).__init__()

        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0)
        return self.conv(x)
    
class DownSampleBlock(nn.Module):
    '''
    Downsamples input to 0.5x and then runs a convolution
    '''
    def __init__(self, channels):
        super(DownSampleBlock, self).__init__()

        self.pad = (0, 1, 0, 1)
        self.conv = nn.Conv2d(channels, channels, 3, 2, 0)

    def forward(self, x):
        x = F.pad(x, self.pad, mode='constant', value=0)

        return self.conv(x)

class NonLocalBlock(nn.Module):
    '''
    Attention Mechanism with Pure Convolutions
    '''

    def __init__(self, channels):
        super(NonLocalBlock, self).__init__()
        self.in_channels = channels

        self.gn = GroupNorm(channels)
        self.q = nn.Conv2d(channels, channels, 1, 1, 0)
        self.k = nn.Conv2d(channels, channels, 1, 1, 0)
        self.v = nn.Conv2d(channels, channels, 1, 1, 0)
        self.proj_out = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        h_ = self.gn(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        batch, channels, height, width = q.shape

        q = q.reshape(batch, channels, height*width)
        q = q.permute(0, 2, 1)
        
        k = k.reshape(batch, channels, height*width)
        v = v.reshape(batch, channels, height*width)

        # A = softmax(Q * K.T / sqrt(d_k)) * V

        attn = torch.bmm(q, k)
        attn = attn * (int(channels) ** (-0.5))
        attn = F.softmax(attn, dim=2)
        attn = attn.permute(0, 2, 1)

        A = torch.bmm(v, attn)
        A = A.reshape(batch, channels, height, width)

        return x + A