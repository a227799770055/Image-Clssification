import torch
import torch.nn as nn
import time
def _make_grid(nx=20, ny=20):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

def translate_to_predict(x, anchors=None):
    if anchors is None:
        anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    else:
        anchors = anchors
    nl = len(anchors)
    na = len(anchors[0]) // 2
    a = torch.tensor(anchors).float().view(nl, -1, 2)
    anchor_grid = a.clone().view(nl, 1, -1, 1, 1, 2).to(x[0].device)
    z = []
    stride = [8, 16, 32]
    for i in range(len(x)):
        bs, na, ny, nx, no = x[i].shape
        grid = _make_grid(nx, ny).to(x[i].device)
        y = x[i].sigmoid()
        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid) * stride[i]  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
        z.append(y.view(bs, -1, no))
    return torch.cat(z, 1)


class YOLOPREDICT(nn.Module):
    def __init__(self, anchros=None):
        super(YOLOPREDICT, self).__init__()
        if anchros is None:
            self.anchros = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        self.stride = [8, 16, 32]
        self.nl = len(self.anchros)
        self.a =  torch.tensor(self.anchros).float().view(self.nl, -1, 2)
    
    def _make_grid(self, nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
    
    def forward(self, x):
        anchor_grid = self.a.clone().view(self.nl, 1, -1, 1, 1, 2).to(x[0].device)
        z = []
        for i in range(len(x)):
            bs, na, ny, nx, no = x[i].shape
            grid = self._make_grid(nx, ny).to(x[i].device)
            y = x[i].sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid) * self.stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
            z.append(y.view(bs, -1, no))
        return torch.cat(z, 1)
