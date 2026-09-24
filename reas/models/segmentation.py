"""Dual-input segmentation network.

Legacy registered layers (block6, up5/db5) remain for checkpoint compatibility.
They are not used by the five-scale forward pass.
"""

import torch
import torch.nn as nn

from .vit import ViTAE


class DoubleConv(nn.Module):
    def __init__(self, in_channels, base_width):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(base_width, base_width, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(base_width)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class EncoderSegment(nn.Module):
    def __init__(
        self,
        in_channels,
        base_width,
        is_double_input=False,
        is_double_attention=False,
        attention_depth=4,
        attention_heads=8,
        patch_size=16,
    ):
        super(EncoderSegment, self).__init__()
        self.double_input = is_double_input
        self.double_atten = is_double_attention
        self.block1 = DoubleConv(in_channels=in_channels, base_width=base_width)
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = DoubleConv(in_channels=base_width, base_width=base_width * 2)
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))
        self.block3 = DoubleConv(in_channels=base_width * 2, base_width=base_width * 4)
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))
        self.block4 = DoubleConv(in_channels=base_width * 4, base_width=base_width * 8)
        self.mp4 = nn.Sequential(nn.MaxPool2d(2))
        self.block5 = DoubleConv(in_channels=base_width * 8, base_width=base_width * 16)
        self.mp5 = nn.Sequential(nn.MaxPool2d(2))
        self.block6 = DoubleConv(in_channels=base_width * 16, base_width=base_width * 32)
        att_embedding = [64, 96, 128, 128, 128]
        num_heads = attention_heads
        self.att1 = ViTAE(
            in_channels=base_width,
            out_channels=base_width,
            patch_size=patch_size,
            embed_dim=att_embedding[0],
            depth=attention_depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
        )
        self.att2 = ViTAE(
            in_channels=base_width * 2,
            out_channels=base_width * 2,
            patch_size=patch_size,
            embed_dim=att_embedding[1],
            depth=attention_depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
        )
        self.att3 = ViTAE(
            in_channels=base_width * 4,
            out_channels=base_width * 4,
            patch_size=patch_size,
            embed_dim=att_embedding[2],
            depth=attention_depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
        )
        self.att4 = ViTAE(
            in_channels=base_width * 8,
            out_channels=base_width * 8,
            patch_size=patch_size,
            embed_dim=att_embedding[3],
            depth=attention_depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
        )
        self.att5 = ViTAE(
            in_channels=base_width * 16,
            out_channels=base_width * 16,
            patch_size=patch_size,
            embed_dim=att_embedding[4],
            depth=attention_depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
        )

    def forward(self, x, x_nor=None, is_double=True):
        if is_double:
            b1_ano = self.block1(x)
            b1_nor = self.block1(x_nor)
            (b1_ano, b1_nor) = self.att1(b1_ano, b1_nor)
            mp1_ano = self.mp1(b1_ano)
            mp1_nor = self.mp1(b1_nor)
            b1 = torch.cat((b1_ano, b1_nor), dim=1)
            b2_ano = self.block2(mp1_ano)
            b2_nor = self.block2(mp1_nor)
            (b2_ano, b2_nor) = self.att2(b2_ano, b2_nor)
            mp2_ano = self.mp2(b2_ano)
            mp2_nor = self.mp2(b2_nor)
            b2 = torch.cat((b2_ano, b2_nor), dim=1)
            b3_ano = self.block3(mp2_ano)
            b3_nor = self.block3(mp2_nor)
            (b3_ano, b3_nor) = self.att3(b3_ano, b3_nor)
            mp3_ano = self.mp3(b3_ano)
            mp3_nor = self.mp3(b3_nor)
            b3 = torch.cat((b3_ano, b3_nor), dim=1)
            b4_ano = self.block4(mp3_ano)
            b4_nor = self.block4(mp3_nor)
            (b4_ano, b4_nor) = self.att4(b4_ano, b4_nor)
            mp4_ano = self.mp4(b4_ano)
            mp4_nor = self.mp4(b4_nor)
            b4 = torch.cat((b4_ano, b4_nor), dim=1)
            b5_ano = self.block5(mp4_ano)
            b5_nor = self.block5(mp4_nor)
            (b5_ano, b5_nor) = self.att5(b5_ano, b5_nor)
            b5 = torch.add(b5_ano, b5_nor)
            return (b1, b2, b3, b4, b5)
        else:
            b1 = self.block1(x)
            b1 = self.att1(b1)
            mp1 = self.mp1(b1)
            b2 = self.block2(mp1)
            b2 = self.att2(b2)
            mp2 = self.mp3(b2)
            b3 = self.block3(mp2)
            b3 = self.att3(b3)
            mp3 = self.mp3(b3)
            b4 = self.block4(mp3)
            b4 = self.att4(b4)
            mp4 = self.mp4(b4)
            b5 = self.block5(mp4)
            b5 = self.att5(b5)
            mp5 = self.mp5(b5)
            b6 = self.block6(mp5)
        return (b1, b2, b3, b4, b5, b6)


class DecoderSegment(nn.Module):
    def __init__(self, base_width=32, out_channels=2, is_double_attention=True):
        super(DecoderSegment, self).__init__()
        concat_layer_num = 3 if is_double_attention else 2
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 16, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )
        self.db1 = nn.Sequential(
            nn.Conv2d(base_width * 8 * concat_layer_num, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
        )
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width * 4 * concat_layer_num, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
        )
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width * 2 * concat_layer_num, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
        )
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        self.db4 = nn.Sequential(
            nn.Conv2d(base_width * 1 * concat_layer_num, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        self.db5 = nn.Sequential(
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        self.fin_out = nn.Sequential(nn.Conv2d(base_width, out_channels, kernel_size=1, padding=0))

    def forward(self, b1, b2, b3, b4, b5):
        up1 = self.up1(b5)
        cat1 = torch.cat((up1, b4), dim=1)
        db1 = self.db1(cat1)
        up2 = self.up2(db1)
        cat2 = torch.cat((up2, b3), dim=1)
        db2 = self.db2(cat2)
        up3 = self.up3(db2)
        cat3 = torch.cat((up3, b2), dim=1)
        db3 = self.db3(cat3)
        up4 = self.up4(db3)
        cat3 = torch.cat((up4, b1), dim=1)
        db4 = self.db4(cat3)
        out = self.fin_out(db4)
        return out


class AttentionSubNetwork(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=2,
        base_width=32,
        attention_depth=4,
        attention_heads=8,
        patch_size=16,
    ):
        super(AttentionSubNetwork, self).__init__()
        self.encoder = EncoderSegment(
            in_channels,
            base_width=base_width,
            is_double_input=True,
            is_double_attention=True,
            attention_depth=attention_depth,
            attention_heads=attention_heads,
            patch_size=patch_size,
        )
        self.decoder = DecoderSegment(
            base_width=base_width, out_channels=out_channels, is_double_attention=True
        )

    def forward(self, x, x_nor=None):
        (b1, b2, b3, b4, b5) = self.encoder(x, x_nor)
        output = self.decoder(b1, b2, b3, b4, b5)
        return output
