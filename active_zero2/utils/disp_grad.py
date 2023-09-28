import torch
import torch.nn as nn


class DispGrad(nn.Module):
    def __init__(self, grad_threshold=100.0):
        super(DispGrad, self).__init__()
        Gx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]])
        Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter = nn.Conv2d(
            in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False, padding_mode="reflect"
        )
        self.filter.weight = nn.Parameter(G, requires_grad=False)
        self.grad_threshold = grad_threshold

    def forward(self, disp):
        assert len(disp.shape) == 4
        disp_grad = self.filter(disp)
        disp_grad = torch.clip(disp_grad, -self.grad_threshold, self.grad_threshold)
        return disp_grad
