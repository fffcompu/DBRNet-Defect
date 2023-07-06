import torch
from torch import nn, Tensor
from torch.nn import functional as F
class AlignedModule(nn.Module):
    def __init__(self, c1, c2, k=3):
        super().__init__()
        self.down_h = nn.Conv2d(c1, c2, 1, bias=False)
        self.down_l = nn.Conv2d(c1, c2, 1, bias=False)
        self.flow_make = nn.Conv2d(c2 * 2, 2, k, 1, 1, bias=False)

    def forward(self, low_feature: Tensor, high_feature: Tensor) -> Tensor:
        high_feature_origin = high_feature
        H, W = low_feature.shape[-2:]
        low_feature = self.down_l(low_feature)
        high_feature = self.down_h(high_feature)
        high_feature = F.interpolate(high_feature, size=(H, W), mode='bilinear', align_corners=True)
        flow = self.flow_make(torch.cat([high_feature, low_feature], dim=1))
        high_feature = self.flow_warp(high_feature_origin, flow, (H, W))
        return high_feature

    def flow_warp(self, x: Tensor, flow: Tensor, size: tuple) -> Tensor:
        # norm = torch.tensor(size).reshape(1, 1, 1, -1)
        norm = torch.tensor([[[[*size]]]]).type_as(x).to(x.device)
        H = torch.linspace(-1.0, 1.0, size[0]).view(-1, 1).repeat(1, size[1])
        W = torch.linspace(-1.0, 1.0, size[1]).repeat(size[0], 1)
        grid = torch.cat((W.unsqueeze(2), H.unsqueeze(2)), dim=2)
        grid = grid.repeat(x.shape[0], 1, 1, 1).type_as(x).to(x.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm
        output = F.grid_sample(x, grid, align_corners=False)
        return output