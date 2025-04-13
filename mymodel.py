import torch
import torch.nn as nn

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=8):
        super(UNet3D, self).__init__()
        
        # Encoder with single convolution at each level
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, features, kernel_size=3, padding=1),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc2 = nn.Sequential(
            nn.Conv3d(features, features*2, kernel_size=3, padding=1),
            nn.BatchNorm3d(features*2),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Bridge
        self.bridge = nn.Sequential(
            nn.Conv3d(features*2, features*4, kernel_size=3, padding=1),
            nn.BatchNorm3d(features*4),
            nn.ReLU(inplace=True)
        )
        
        # Decoder with single convolution at each level
        self.up2 = nn.ConvTranspose3d(features*4, features*2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv3d(features*4, features*2, kernel_size=3, padding=1),
            nn.BatchNorm3d(features*2),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.ConvTranspose3d(features*2, features, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv3d(features*2, features, kernel_size=3, padding=1),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True)
        )
        
        self.final = nn.Conv3d(features, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Save input size for debugging
        input_size = x.size()
        
        # Encoder
        e1 = self.enc1(x)
        e1_size = e1.size()  # Save size for debugging
        x = self.pool1(e1)
        
        e2 = self.enc2(x)
        e2_size = e2.size()  # Save size for debugging
        x = self.pool2(e2)
        
        # Bridge
        x = self.bridge(x)
        bridge_size = x.size()  # Save size for debugging
        
        # Decoder
        x = self.up2(x)
        up2_size = x.size()  # Save size for debugging
        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x)
        
        x = self.up1(x)
        up1_size = x.size()  # Save size for debugging
        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x)
        
        # Print sizes for debugging
        print(f"Input size: {input_size}")
        print(f"Encoder 1 size: {e1_size}")
        print(f"Encoder 2 size: {e2_size}")
        print(f"Bridge size: {bridge_size}")
        print(f"Up2 size: {up2_size}")
        print(f"Up1 size: {up1_size}")
        
        x = self.final(x)
        x = self.sigmoid(x)
        
        return x

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, predictions, targets):
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice
