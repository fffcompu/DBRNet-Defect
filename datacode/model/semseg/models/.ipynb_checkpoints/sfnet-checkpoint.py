import os
import sys
sys.path.append('/root/autodl-tmp/MyFrame/datacode/model')
import torch
from torch import Tensor
from torch.nn import functional as F
from semseg.models.base import BaseModel
from semseg.models.heads import SFHead
from torchstat import stat
import time


class SFNet(BaseModel):
    def __init__(self, backbone: str = 'ResNetD-18', num_classes: int = 19):
        assert 'ResNet' in backbone
        super().__init__(backbone, num_classes)
        self.head = SFHead(self.backbone.channels, 128 if '18' in backbone else 256, num_classes)
        self.apply(self._init_weights)

    def forward(self, x: Tensor) -> Tensor:
        outs = self.backbone(x)
        out = self.head(outs)
        out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=True)
        return out


if __name__ == '__main__':


#    #model.init_pretrained(r'C:\Users\86157\.cache\torch\hub\checkpoints\resnet18-5c106cde.pth')
#    x = torch.randn(2, 3, 224, 224)
#    y = model(x)
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    model = SFNet('ResNet-18',num_classes=4).cuda()
    model.train(False)
    model.eval()

    input=torch.randn(1,3,1024,2048).cuda()
    warm_iter=40
    iteration=250
    print('=========Speed Testing=========')
    fps_time=[]
    for _ in range(iteration): #iteration=20
        if _<warm_iter:
            model(input)
        else:
            torch.cuda.synchronize()
            start=time.time()
            model(input)
            torch.cuda.synchronize()
            end=time.time()
            fps_time.append(end-start)
            print(end-start)
    time_sum = 0
    for i in fps_time:
        time_sum += i
    print("FPS: %f"%(1.0/(time_sum/len(fps_time))))
