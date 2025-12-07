import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """
        Attention Gate to suppress irrelevant background noise.
        """
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class AttentionResUNet(nn.Module):
    def __init__(self, in_channels=9, out_channels=1):
        super().__init__()
        
        # 1. Encoder (Pre-trained ResNet34)
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        
        # Modify first layer for N channels (Weight Averaging)
        if in_channels != 3:
            original_conv = resnet.conv1
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                resnet.conv1.weight[:, :3] = original_conv.weight
                for i in range(3, in_channels):
                    resnet.conv1.weight[:, i] = original_conv.weight.mean(dim=1)

        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.enc1 = resnet.layer1 # 64
        self.enc2 = resnet.layer2 # 128
        self.enc3 = resnet.layer3 # 256
        self.enc4 = resnet.layer4 # 512 (Bottleneck)
        
        # 2. Attention Gates
        # F_g must match the Upsampled Decoder feature size
        self.att3 = AttentionBlock(F_g=512, F_l=256, F_int=128)
        self.att2 = AttentionBlock(F_g=256, F_l=128, F_int=64)
        self.att1 = AttentionBlock(F_g=128, F_l=64,  F_int=32)
        
        # 3. Decoder
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = ConvBlock(512 + 256, 256)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = ConvBlock(256 + 128, 128)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = ConvBlock(128 + 64, 64)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = ConvBlock(64 + 64, 32)
        
        # Aux Heads
        self.aux_head = nn.Conv2d(128, out_channels, kernel_size=1)
        self.global_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0 = self.enc0(x)
        e1 = self.enc1(x0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        global_pred = self.global_head(e4)
        
        # Decoder with Attention
        d4 = self.up4(e4)
        e3_att = self.att3(g=d4, x=e3)
        d4 = torch.cat([d4, e3_att], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        e2_att = self.att2(g=d3, x=e2)
        d3 = torch.cat([d3, e2_att], dim=1)
        d3 = self.dec3(d3)
        
        aux_seg = self.aux_head(d3)
        
        d2 = self.up2(d3)
        e1_att = self.att1(g=d2, x=e1)
        d2 = torch.cat([d2, e1_att], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        # Simple skip for first layer
        e1_resized = F.interpolate(e1, size=d1.shape[2:], mode='bilinear', align_corners=True)
        d1 = torch.cat([d1, e1_resized], dim=1)
        d1 = self.dec1(d1)
        
        d1 = F.interpolate(d1, size=x.size()[2:], mode='bilinear', align_corners=True)
        seg_final = self.final_conv(d1)
        
        return seg_final, aux_seg, global_pred