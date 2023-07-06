import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchstat import stat
import sys
from Alignedmodule import AlignedModule
from DCPPM import DCPPM
import os
import time
class FastDownSample(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, groups=c1,bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True),
            nn.Conv2d(c2,c2,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(c2)

        )

class FastDownSample2d(nn.Sequential):
    def __init__(self, c1, ch, c2, k, s=1, p=0):
        super().__init__(
            nn.Conv2d(c1, ch, k, s, p, groups=c1,bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(True),
            nn.Conv2d(ch,ch,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(ch),
            nn.Conv2d(ch, c2, k, s, p, bias=False,groups=ch),
            nn.BatchNorm2d(c2),
            nn.ReLU(True),
            nn.Conv2d(c2, c2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(c2),

        )
class BasicBlockOriginal(nn.Module):
    expansion = 1
    def __init__(self, c1, c2, s=1, downsample= None, no_relu=False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2, 3, s, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.conv2 = nn.Conv2d(c2, c2, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        return out if self.no_relu else F.relu(out)
class LFEM(nn.Module):
    expansion = 1
    middleExpansion = 2

    def __init__(self, c1, c2, s=1, downsample=None, no_relu=False) -> None:
        super().__init__()
        self.conv2_last = None
        if s == 2:
            self.conv1 = nn.Conv2d(c1, c2 * self.middleExpansion, 1, 1, 0, bias=False)
            self.bn1 = nn.BatchNorm2d(c2 * self.middleExpansion)
            self.conv2 = nn.Conv2d(c2 * self.middleExpansion, c2 * self.middleExpansion, 3, s, 1,
                                   groups=c2 * self.middleExpansion,
                                   bias=False)
            self.bn2 = nn.BatchNorm2d(c2 * self.middleExpansion)

            self.conv3 = nn.Conv2d(c2 * self.middleExpansion, c2, 1, 1, 0)
            self.bn3 = nn.BatchNorm2d(c2)
            self.downsample = downsample
            self.no_relu = no_relu
        elif s == 1:
            self.conv1 = nn.Conv2d(c1, c2 * self.middleExpansion, 1, 1, 0, bias=False)
            self.bn1 = nn.BatchNorm2d(c2 * self.middleExpansion)

            #3*3的dw卷积

            self.conv2 = nn.Conv2d(c2 , c2, 3, s, 1,
                                   groups=c2,
                                   bias=False)
            self.bn2 = nn.BatchNorm2d(c2)

            # 中间加入的5*5dw卷积
            self.conv2_last = nn.Conv2d(c2 , c2, 5, s, 2,
                                        groups=c2,
                                        bias=False)
            self.bn2_last_bn = nn.BatchNorm2d(c2)

            self.conv3 = nn.Conv2d(c2 * self.middleExpansion, c2, 1, 1, 0)
            self.bn3 = nn.BatchNorm2d(c2)
            self.downsample = downsample
            self.no_relu = no_relu

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        if self.conv2_last is not None:
            x1, x2 = out.chunk(2, dim=1)
            out1 = self.bn2(self.conv2(x1))
            out2 = self.bn2_last_bn(self.conv2_last(x2))
            out = torch.concat((out1,out2),1)
            out=F.relu(out)
        else:
            out1 = self.bn2(self.conv2(out))
            out = F.relu(out1)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        if  self.conv2_last is not None:
            out = channel_shuffle(out, 2)
        return out if self.no_relu else F.relu(out)

def channel_shuffle(x: Tensor, groups: int) -> Tensor:

    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, c1, c2, s=1, downsample=None, no_relu=False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.conv2 = nn.Conv2d(c2, c2, 3, s, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.conv3 = nn.Conv2d(c2, c2 * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(c2 * self.expansion)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        return out if self.no_relu else F.relu(out)


class ConvBN(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, bias=False),
            nn.BatchNorm2d(c2)
        )


class Conv2BN(nn.Sequential):
    def __init__(self, c1, ch, c2, k, s=1, p=0):
        super().__init__(
            nn.Conv2d(c1, ch, k, s, p, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(True),
            nn.Conv2d(ch, c2, k, s, p, bias=False),
            nn.BatchNorm2d(c2)
        )


class Stem(nn.Sequential):
    def __init__(self, c1, c2):
        super().__init__(
            nn.Conv2d(c1, c2, 3, 2, 1),
            nn.BatchNorm2d(c2),
            nn.ReLU(True),
            nn.Conv2d(c2, c2, 3, 2, 1),
            nn.BatchNorm2d(c2),
            nn.ReLU(True)
        )


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
class SegHead(nn.Module):
    def __init__(self, c1, ch, c2, scale_factor=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(c1)
        self.conv1 = nn.Conv2d(c1, ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, c2, 1)
        self.scale_factor = scale_factor

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(x)))

        if self.scale_factor is not None:
            H, W = x.shape[-2] * self.scale_factor, x.shape[-1] * self.scale_factor
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out


class DBRNet(nn.Module):
    def __init__(self, backbone: str = None, num_classes: int = 19) -> None:
        super().__init__()
        planes, spp_planes, head_planes = [32, 64, 128, 256, 512], 128, 64
        self.conv1 = Stem(3, planes[0])

        self.layer1 = self._make_layer(BasicBlockOriginal, planes[0], planes[0], 2)
        self.layer2 = self._make_layer(LFEM, planes[0], planes[1], 2, 2)
        self.layer3 = self._make_layer(BasicBlockOriginal, planes[1], planes[2], 2, 2)
        self.layer4 = self._make_layer(LFEM, planes[2], planes[3], 2, 2)
        self.layer5 = self._make_layer(Bottleneck, planes[3], planes[3], 1)

        self.layer3_ = self._make_layer(BasicBlockOriginal, planes[1], planes[1], 1)
        self.layer4_ = self._make_layer(BasicBlockOriginal, planes[1], planes[1], 1)
        self.layer5_ = self._make_layer(Bottleneck, planes[1], planes[1], 1)

        self.alignlayer3_ = AlignedModule(planes[1], planes[1] // 2)

        self.compression3 = ConvBN(planes[2], planes[1], 1)
        self.compression4 = ConvBN(planes[3], planes[1], 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.wavg1=nn.Conv2d(planes[2],planes[1],kernel_size=1,padding=0)
        #self.wavg2=nn.Conv2d(planes[3],planes[1],kernel_size=1,padding=0)

        self.down3 = ConvBN(planes[1], planes[2], 3, 2, 1)
        self.down4 = Conv2BN(planes[1], planes[2], planes[3], 3, 2, 1)

        self.spp = DCPPM(planes[-1], spp_planes, planes[2])
        self.seghead_extra = SegHead(planes[1], head_planes, num_classes, 8)
        self.final_layer = SegHead(planes[2], head_planes, num_classes, 8)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            self.load_state_dict(torch.load(pretrained, map_location='cpu')['model'], strict=False)

    def _make_layer(self, block, inplanes, planes, depths, s=1) -> nn.Sequential:
        downsample = None
        if inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, 1, s, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = [block(inplanes, planes, s, downsample)]
        inplanes = planes * block.expansion

        for i in range(1, depths):
            if i == depths - 1:
                layers.append(block(inplanes, planes, no_relu=True))
            else:
                layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)
    def _make_layer_original(self, block, inplanes, planes, depths, s=1) -> nn.Sequential:
        downsample = None
        if inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, 1, s, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = [block(inplanes, planes, s, downsample)]
        inplanes = planes * block.expansion

        for i in range(1, depths):
            if i == depths - 1:
                layers.append(block(inplanes, planes, no_relu=True))
            else:
                layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        H, W = x.shape[-2] // 8, x.shape[-1] // 8
        layers = []

        x = self.conv1(x)
        x = self.layer1(x)
        layers.append(x)

        x = self.layer2(F.relu(x))
        layers.append(x)

        x = self.layer3(F.relu(x))
        layers.append(x)
        x_ = self.layer3_(F.relu(layers[1]))
        x = x + self.down3(F.relu(x_))
        #AFFM融合模块
        templayer3 = F.relu(layers[2])
        # 高级语义信息生成的权重
        weight1 = self.wavg1(self.avg_pool(templayer3))
        compressionlayer3 = self.compression3(templayer3)
        x_ = weight1 * x_ + self.alignlayer3_(x_, compressionlayer3)
        if self.training: x_aux = self.seghead_extra(x_)
        x = self.layer4(F.relu(x))
        layers.append(x)
        x_ = self.layer4_(F.relu(x_))
        x = x + self.down4(F.relu(x_))
        x_ = self.layer5_(F.relu(x_))
        x = F.interpolate(self.spp(self.layer5(F.relu(x))), size=(H, W), mode='bilinear', align_corners=False)
        x_ = self.final_layer(x + x_)

        return (x_, x_aux) if self.training else x_


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    model= DBRNet(num_classes=5).cuda()
    print(model)
    model.train(False)
    model.eval()
    input=torch.randn(1,3,1600,256).cuda()
    warm_iter=300
    iteration=1000
    print('=========Speed Testing=========')
    fps_time=[]
    for _ in range(iteration): #iteration=20
        if _<warm_iter:
            model(input)
        else:
            torch.cuda.synchronize()
            start=time.time()
            output=model(input)
            torch.cuda.synchronize()
            end=time.time()
            fps_time.append(end-start)
            print(end-start)
    time_sum = 0
    for i in fps_time:
        time_sum += i
    print("FPS: %f"%(1.0/(time_sum/len(fps_time))))