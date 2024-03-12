import torch.nn as nn
import torch
import torch.nn.functional as F

class ConvolutionBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvolutionBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x

class UpConvolutionBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConvolutionBlock, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, H):
        super(AttentionBlock, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, H, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(H)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, H, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(H)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(H, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

class AttentionU_Net(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(AttentionU_Net, self).__init__()

        n_filter = 64

        self.MaxPool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.MaxPool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.MaxPool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.MaxPool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvolutionBlock(in_ch, n_filter)
        self.Conv2 = ConvolutionBlock(n_filter, 2*n_filter)
        self.Conv3 = ConvolutionBlock(2*n_filter, 4*n_filter)
        self.Conv4 = ConvolutionBlock(4*n_filter, 8*n_filter)
        self.Conv5 = ConvolutionBlock(8*n_filter, 16*n_filter)

        self.Up4 = UpConvolutionBlock(16*n_filter, 8*n_filter)
        self.Up3 = UpConvolutionBlock(8*n_filter, 4*n_filter)
        self.Up2 = UpConvolutionBlock(4*n_filter, 2*n_filter)
        self.Up1 = UpConvolutionBlock(2*n_filter, n_filter)

        self.Attention4 = AttentionBlock(F_g=8*n_filter, F_l=8*n_filter, H=4*n_filter)
        self.Attention3 = AttentionBlock(F_g=4*n_filter, F_l=4*n_filter, H=2*n_filter)
        self.Attention2 = AttentionBlock(F_g=2*n_filter, F_l=2*n_filter, H=n_filter)
        self.Attention1 = AttentionBlock(F_g=n_filter, F_l=n_filter, H=n_filter//2)

        self.UpConv4 = ConvolutionBlock(16*n_filter, 8*n_filter)
        self.UpConv3 = ConvolutionBlock(8*n_filter, 4*n_filter)
        self.UpConv2 = ConvolutionBlock(4*n_filter, 2*n_filter)
        self.UpConv1 = ConvolutionBlock(2*n_filter, n_filter)

        self.Conv = nn.Conv2d(n_filter, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        cat = self.Conv1(x)

        cat2 = self.MaxPool1(cat)
        cat2 = self.Conv2(cat2)

        cat3 = self.MaxPool2(cat2)
        cat3 = self.Conv3(cat3)

        cat4 = self.MaxPool3(cat3)
        cat4 = self.Conv4(cat4)

        out = self.MaxPool4(cat4)
        out = self.Conv5(out)

        # -------------------------------

        out = self.Up4(out)
        att = self.Attention4(g=out, x=cat4)
        out = torch.cat((att, out), dim=1)
        out = self.UpConv4(out)

        out = self.Up3(out)
        att = self.Attention3(g=out, x=cat3)
        out = torch.cat((att, out), dim=1)
        out = self.UpConv3(out)

        out = self.Up2(out)
        att = self.Attention2(g=out, x=cat2)
        out = torch.cat((att, out), dim=1)
        out = self.UpConv2(out)

        out = self.Up1(out)
        att = self.Attention1(g=out, x=cat)
        out = torch.cat((att, out), dim=1)
        out = self.UpConv1(out)

        out = self.Conv(out)

        return out

class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss,self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

