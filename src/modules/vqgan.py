import torch.nn as nn
from vqgan_helpers import ResidualBlock, NonLocalBlock, DownSampleBlock, UpSampleBlock, GroupNorm, Swish

class VQGANEncoder(nn.Module):
    def __init__(self, args):
        super(VQGANEncoder, self).__init__()

        channels = [128, 128, 128, 256, 256, 512] # TODO Parameterize
        attn_resolutions = [16]

        num_res_blocks = 2
        resolution = 256 # TODO remove hard code -> Parameterize

        layers = [nn.Conv2d(args.image_channels, channels[0], 3, 1, 1)]

        for i in range(len(channels) - 1):
            in_channels = channels[i]
            out_channels = channels[i + 1]

            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels

                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
                
            if i != len(channels) - 2:
                layers.append(DownSampleBlock(channels[i + 1]))
                resolution //= 2
            
        layers.extend([
            ResidualBlock(channels[-1], channels[-1]),
            NonLocalBlock(channels[-1]),
            ResidualBlock(channels[-1], channels[-1]),
            GroupNorm(channels[-1]),
            Swish(),
            nn.Conv2d(channels[-1], args.latent_dim, 3, 1, 1)
        ])

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class VQGANDecoder(nn.Module):
    def __init__(self, args):
        super(VQGANDecoder, self).__init__()

        channels = [512, 256, 256, 128, 128, 128] # TODO Parameterize
        attn_resolutions = [16]

        num_res_blocks = 3
        resolution = 16

        in_channels = channels[0]
        layers = [
            nn.Conv2d(args.latent_dim, in_channels, 3, 1, 1),
            ResidualBlock(in_channels, in_channels),
            NonLocalBlock(in_channels),
            ResidualBlock(in_channels, in_channels)
        ]

        for i in range(len(channels)):
            out_channels = channels[i]

            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels

                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
                
            if i != 0:
                layers.append(UpSampleBlock(in_channels))
                resolution *= 2
            
        layers.extend([
            GroupNorm(in_channels),
            Swish(),
            nn.Conv2d(in_channels, args.image_channels, 3, 1, 1)
        ])

        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)