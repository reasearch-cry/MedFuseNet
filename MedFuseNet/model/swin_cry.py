import torch
import torchvision
import torch.nn as nn
import logging
from model.ASPP_module import ASPP
from model.Swin_Transformer import SwinTransformer, PatchEmbedding
from model.croattention import CrossAttention
from model.Segmentation_Head import SegmentationHead
from einops.layers.torch import Rearrange
from utils_hiformer import PatchMerging
from timm.models.layers import trunc_normal_
import os
import wget
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)


class cry(nn.Module):
    def __init__(self, img_size, num_classes):
        super(cry, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.cconv1 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=1)
        self.cconv2 = nn.Conv2d(in_channels=192, out_channels=512, kernel_size=1)
        self.cconv3 = nn.Conv2d(in_channels=384, out_channels=1024, kernel_size=1)
        self.aspp = ASPP(in_channel=256)
        self.relu = nn.ReLU()
        # self.conv2=nn.Conv2d(in_channels=3,out_channels=3,kernel_size=3,stride=1,padding=1)
        # self.conv3=nn.Conv2d(in_channels=3,out_channels=3,kernel_size=3,stride=1,padding=1)
        self.conv4 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1)
        # self.gap=nn.AdaptiveAvgPool2d(1)
        resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet = nn.ModuleList(resnet.children())[:7]
        self.norm_1 = nn.LayerNorm(96)

        self.p1_pm = PatchMerging((56, 56), 96)
        self.p2_pm = PatchMerging((28, 28), 192)

        self.patchembed = PatchEmbedding(in_channels=3, out_channels=96)

        if not os.path.isfile('/root/cry/weights/swin_tiny_patch4_window7_224.pth'):
            print('Downloading Swin-transformer model ...')
            wget.download(
                "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",
                "/root/cry/weights/swin_tiny_patch4_window7_224.pth")
        swin_pretrained_path = '/root/cry/weights/swin_tiny_patch4_window7_224.pth'


        model_path = swin_pretrained_path
        self.swin_transformer = SwinTransformer(img_size,in_chans = 3)
        checkpoint = torch.load(model_path, map_location=torch.device(device))['model']
        unexpected = ["patch_embed.proj.weight", "patch_embed.proj.bias", "patch_embed.norm.weight", "patch_embed.norm.bias",
                     "head.weight", "head.bias", "layers.0.downsample.norm.weight", "layers.0.downsample.norm.bias",
                     "layers.0.downsample.reduction.weight", "layers.1.downsample.norm.weight", "layers.1.downsample.norm.bias",
                     "layers.1.downsample.reduction.weight", "layers.2.downsample.norm.weight", "layers.2.downsample.norm.bias",
                     "layers.2.downsample.reduction.weight", "norm.weight", "norm.bias"]


        for key in list(checkpoint.keys()):
            if key in unexpected or 'layers.3' in key:
                del checkpoint[key]
        self.swin_transformer.load_state_dict(checkpoint)


        # self.swin_transformer = SwinTransformer(img_size, in_chans=3)
        self.crossattention = CrossAttention(1024)

        self.deconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.groupnorm1 = nn.GroupNorm(128, 128)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        self.deconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.groupnorm2 = nn.GroupNorm(256, 256)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.deconv3 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.groupnorm3 = nn.GroupNorm(512, 512)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.segmentation_head = SegmentationHead(in_channels=128, out_channels=9, kernel_size=3)

        self.avgpool_1 = nn.AdaptiveAvgPool1d(1)
        self.avgpool_2 = nn.AdaptiveAvgPool1d(1)

        # self.avgpool_e1=nn.AdaptiveAvgPool1d(1)
        self.crossattentione1 = CrossAttention(256)
        self.crossattentione2 = CrossAttention(512)
        self.crossattentione3 = CrossAttention(1024)


        self.sebegin = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 512, kernel_size=1),
            nn.Sigmoid()
        )
        self.sebegin1=nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(1024,256,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256,1024,kernel_size=1),
            nn.Sigmoid()
        )

        self.pos_embed1 = nn.Parameter(torch.zeros(1, 196, 1024))
        if self.pos_embed1.requires_grad:
            trunc_normal_(self.pos_embed1, std=.02)

        self.pos_embed2 = nn.Parameter(torch.zeros(1, 196, 1024))
        if self.pos_embed2.requires_grad:
            trunc_normal_(self.pos_embed2, std=.02)

    def forward(self, x):
        # print(self.swin_transformer.layers[0])
        # print(self.swin_transformer.layers[1])
        # print(self.swin_transformer.layers[2])
        
        x11 = self.conv(x)

        for i in range(5):
            x11 = self.resnet[i](x11)
        x11 = x11

        x21 = self.conv(x)
        x21 = self.patchembed(x21)
        x21 = Rearrange('b c h w -> b (h w) c')(x21)
        x21go = x21 = self.swin_transformer.layers[0](x21)
        # x21=self.norm_1(x21)
        x21 = x21.permute([0, 2, 1])
        B = x21.shape[0]
        x21 = x21.reshape([B, 96, 56, 56])
        x21 = self.cconv1(x21)

        # half-cross attention
        # x21=x21.reshape([B,256,56*56])
        # # # # x21=self.avgpool_e1(x21)
        # x21 = x21.permute([0, 2, 1])
        # xcross=self.crossattentione1(x21)
        # x11special=x11.reshape([B,256,56*56])
        # x11special=x11special.permute([0,2,1])
        # xneedadd=x11special[:,1:,:]
        # xgross=torch.cat((xcross,xneedadd),dim=1)
        # xgross=xgross.reshape([B,56,56,256])
        # xgross=xgross.permute([0,3,1,2])
        # x31=x11+xgross

        x11special = x11.reshape([B, 256, 56 * 56])
        x11special = x11special.permute([0, 2, 1])
        x11special = self.crossattentione1(x11special)
        x21 = x21.reshape([B, 256, 56 * 56])
        x21 = x21.permute([0, 2, 1])
        x1gross = torch.cat((x11special, x21[:, 1:, :]), dim=1)
        x1gross = x1gross.permute([0, 2, 1])
        x1gross = x1gross.reshape([B, 256, 56, 56])
        x31 = x1gross + x11

        x12 = self.aspp(x11)
        x12 = self.relu(x12)

        x12 = self.resnet[5](x12)
        x21go = self.p1_pm(x21go)
        x22go = x22 = self.swin_transformer.layers[1](x21go)
        # x13=self.conv2(x12) #24 3 224 224
        # x13=self.relu(x13)
        x22 = x22.permute([0, 2, 1])
        x22 = x22.reshape([B, 192, 28, 28])
        x22 = self.cconv2(x22)

        x12pro = x12.reshape([B, 512, 28 * 28])
        x12pro = x12pro.permute([0, 2, 1])
        x12pro = self.crossattentione2(x12pro)
        x22 = x22.reshape([B, 512, 28 * 28])
        x22 = x22.permute([0, 2, 1])
        x2gross = torch.cat((x12pro, x22[:, 1:, :]), dim=1)
        x2gross = x2gross.permute([0, 2, 1])
        x2gross = x2gross.reshape([B, 512, 28, 28])

        x32 = x12 + x2gross

        # x121=self.se1(x12)
        # x12=x12*x121
        #
        # #SE attention
        # x22go=x22go.permute([0,2,1])
        # x22go=x22go.reshape([B,192,28,28])
        # x22golater=self.se2(x22go)
        # x22go=x22go*x22golater
        # x22go=x22go.reshape([B,192,28*28])
        # x22go=x22go.permute([0,2,1])

        x13 = self.resnet[6](x12)
        # x14=self.conv3(x13) #torch.Size([24, 3, 224, 224])
        x22go = self.p2_pm(x22go)

        # x222=self.se2(x22go)
        # x22go=x22go*x222

        x23go = x23 = self.swin_transformer.layers[2](x22go)
        x23 = x23.permute([0, 2, 1])
        x23 = x23.reshape([B, 384, 14, 14])
        x23 = self.cconv3(x23)

        x13pro = x13.reshape([B, 1024, 14 * 14])
        x13pro = x13pro.permute([0, 2, 1])
        x13pro = self.crossattentione3(x13pro)
        x23pro = x23
        x23pro = x23pro.reshape([B, 1024, 14 * 14])
        x23pro = x23pro.permute([0, 2, 1])
        x13pro = torch.cat((x13pro, x23pro[:, 1:, :]), dim=1)
        x13pro = x13pro.permute([0, 2, 1])
        x13pro = x13pro.reshape([B, 1024, 14, 14])

        x33 = x13 + x13pro

        # xgap1=self.gap(x14) #torch.Size([24, 3, 1, 1])
        # xgap2=self.gap(x23)

        B, C, H, W = x13.shape
        N = H * W
        x13 = x13.reshape([B, N, C])

        # position:
        x13 = self.pos_embed1 + x13

        x13one = x13.permute([0, 2, 1])
        # x13one=self.avgpool_1(x13one)
        x13one = x13one.permute([0, 2, 1])

        xdeficient1 = self.crossattention(x13one)
        x23 = x23.reshape([B, N, C])
        xsupplement1 = x23[:, 1:, ...]
        xultimate1 = torch.cat((xdeficient1, xsupplement1), dim=1)

        x23 = x23.reshape([B, N, C])
        x23 = x23 + self.pos_embed2
        x23two = x23.permute([0, 2, 1])
        # x23two=self.avgpool_2(x23two)
        x23two = x23two.permute([0, 2, 1])
        xdeficient2 = self.crossattention(x23two)
        x14 = x13
        xsupplement2 = x14[:, 1:, ...]
        xultimate2 = torch.cat((xdeficient2, xsupplement2), dim=1)

        xultimate = xultimate1 + xultimate2
        xultimate = xultimate.reshape(B, C, H, W)

        xultimatepro=self.sebegin1(xultimate)
        xultimate3=xultimatepro*xultimate
        
        xultimate3 = self.deconv3(xultimate3 + x33)
        xultimate3 = self.groupnorm3(xultimate3)
        xultimate3 = self.relu(xultimate3)
        xultimate3 = self.upsample3(xultimate3)

        # xultimate3pro = self.sebegin(xultimate3)
        # xultimate3 = xultimate3pro * xultimate3

        xultimate2 = self.deconv2(xultimate3 + x32)
        xultimate2 = self.groupnorm2(xultimate2)
        xultimate2 = self.relu(xultimate2)
        xultimate2 = self.upsample2(xultimate2)

        xultimate1 = self.deconv1(xultimate2 + x31)
        xultimate1 = self.groupnorm1(xultimate1)
        xultimate1 = self.relu(xultimate1)

        #         xultimate1pro=self.seend(xultimate1)
        #         xultimate1=xultimate1pro*xultimate1

        xultimate1 = self.upsample1(xultimate1)

        xultimate = self.segmentation_head(xultimate1)

        return xultimate


if __name__ == '__main__':
    cry1 = cry(224, 4)
    input = torch.rand(4, 1, 224, 224)
    print(cry1(input).shape)
#     cry2=cry(224,4)
#     input=torch.rand(24,1,224,224)
#     print(cry2(input).shape)
#     cry3=cry(224,4)
#     input=torch.rand(24,1,224,224)
#     print(cry3(input).shape)