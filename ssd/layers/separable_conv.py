from torch import nn


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, onnx_compatible=False):
        super().__init__()
        ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                      groups=in_channels, stride=stride, padding=padding),
            nn.BatchNorm2d(in_channels),
            ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.conv(x)
