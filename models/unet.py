"""
Lightweight U-Net Convolutional Neural Network for Biomedical Image Segmentation.
Implements encoder-decoder architecture with skip connections.
"""
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

if torch is not None:
    class DoubleConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            return self.net(x)

    class UNet(nn.Module):
        def __init__(self, in_channels=1, num_classes=1):
            super().__init__()
            # Contracting path (Encoder)
            self.inc = DoubleConv(in_channels, 16)
            self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(16, 32))
            self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))

            # Bottleneck
            self.bottleneck = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))

            # Expansive path (Decoder)
            self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.conv_up1 = DoubleConv(128, 64)

            self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
            self.conv_up2 = DoubleConv(64, 32)

            self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
            self.conv_up3 = DoubleConv(32, 16)

            self.outc = nn.Conv2d(16, num_classes, kernel_size=1)

        def forward(self, x):
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            xb = self.bottleneck(x3)

            x = self.up1(xb)
            x = self.conv_up1(torch.cat([x, x3], dim=1))

            x = self.up2(x)
            x = self.conv_up2(torch.cat([x, x2], dim=1))

            x = self.up3(x)
            x = self.conv_up3(torch.cat([x, x1], dim=1))

            return torch.sigmoid(self.outc(x))
else:
    class UNet:
        pass
