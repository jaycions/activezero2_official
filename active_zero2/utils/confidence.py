import torch


def compute_confidence(disp_pred_norm, prob_volume, margin=4) -> torch.Tensor:
    B, D, H, W = prob_volume.shape
    grid = torch.linspace(0, 1, D).view(1, D, 1, 1).expand(B, D, H, W).to(prob_volume.device)
    confs = []
    with torch.no_grad():
        for m in range(1, margin + 1):
            disp_pred_floor = disp_pred_norm - m / (D - 1)
            disp_pred_floor = torch.clip(disp_pred_floor, 0, 1)
            disp_pred_ceil = disp_pred_norm + m / (D - 1)
            disp_pred_ceil = torch.clip(disp_pred_ceil, 0, 1)
            mask = (grid >= disp_pred_floor) * (grid <= disp_pred_ceil)
            conf = torch.sum(mask * prob_volume, 1, keepdim=True)
            confs.append(conf)

        confs = torch.cat(confs, dim=1)
    return confs
