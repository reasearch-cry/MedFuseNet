from torch import nn as nn

class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,stride=1,padding=1)
        super().__init__(conv2d)