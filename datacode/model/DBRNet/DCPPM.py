import torch
from torch import nn, Tensor
from torch.nn import functional as F
class Scale(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0):
        super().__init__(
            nn.AvgPool2d(k, s, p),
            nn.BatchNorm2d(c1),
            nn.ReLU(True),
            nn.Conv2d(c1, c2, 1, bias=False)
        )


class ScaleLast(nn.Sequential):
    def __init__(self, c1, c2, k):
        super().__init__(
            nn.AdaptiveAvgPool2d(k),
            nn.BatchNorm2d(c1),
            nn.ReLU(True),
            nn.Conv2d(c1, c2, 1, bias=False)
        )


class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0):
        super().__init__(
            nn.BatchNorm2d(c1),
            nn.ReLU(True),
            nn.Conv2d(c1, c2, k, s, p, bias=False)
        )
class DWConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0):
        super().__init__(
            nn.BatchNorm2d(c1),
            nn.ReLU(True),
            nn.Conv2d(c1, c2, k, s, p,groups=c1, bias=False)
        )


class DCPPM(nn.Module):
    def __init__(self, c1, ch, c2):
        super().__init__()
        self.scale1 = Scale(c1, ch, 5, 2, 2)
        self.scale2 = Scale(c1, ch, 9, 4, 4)
        self.scale3 = Scale(c1, ch, 17, 8, 8)
        self.scale4 = ScaleLast(c1, ch, 1)
        self.scale0 = ConvModule(c1, ch, 1)
        #将ConvModule更改为DWConvModule
        self.process1 = DWConvModule(ch, ch, 3, 1, 1)
        self.process2 = DWConvModule(ch, ch, 3, 1, 1)
        self.process3 = DWConvModule(ch, ch, 3, 1, 1)
        self.process4 = DWConvModule(ch, ch, 3, 1, 1)
        self.compression = ConvModule(ch * 5, c2, 1)
        self.shortcut = ConvModule(c1, c2, 1)

    def forward(self, x: Tensor) -> Tensor:
        outs = [self.scale0(x)]

        outUnsamplesSale1 = F.interpolate(self.scale1(x), size=x.shape[-2:], mode='bilinear', align_corners=False)
        outUnsamplesSale2 = F.interpolate(self.scale2(x), size=x.shape[-2:], mode='bilinear', align_corners=False)
        outUnsamplesSale3 = F.interpolate(self.scale3(x), size=x.shape[-2:], mode='bilinear', align_corners=False)
        outUnsamplesSale4 = F.interpolate(self.scale4(x), size=x.shape[-2:], mode='bilinear', align_corners=False)

        outProcess1 = self.process1(outUnsamplesSale1 + outs[-1])
        outs.append(outProcess1)
        addBranch2_1 = outUnsamplesSale2 + outs[-1]
        outProcess2 = self.process2(addBranch2_1)
        outs.append(outProcess2)
        addBranch3_1 = outUnsamplesSale3 + addBranch2_1
        addBranch3_2 = addBranch3_1 + outs[-1]
        outProcess3 = self.process3(addBranch3_2)
        outs.append(outProcess3)
        addBranch4_1 = outUnsamplesSale4 + addBranch3_1
        addBranch4_2 = addBranch4_1 + addBranch3_2
        addBranch4_3 = addBranch4_2 + outs[-1]
        outProcess4 = self.process4(addBranch4_3)
        outs.append(outProcess4)
        out = self.compression(torch.cat(outs, dim=1)) + self.shortcut(x)
        return out
if __name__ == '__main__':
    spp = DCPPM(512, 128, 128)
    input = torch.randn(2, 512, 7, 7)
    output = spp(input)
    print(spp)
