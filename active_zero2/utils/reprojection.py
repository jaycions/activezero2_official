"""
Author: Isabella Liu 10/5/21
Feature:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_disparity(img, disp):
    batch_size, _, height, width = img.size()
    disp = disp / width

    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, width).repeat(batch_size, height, 1).type_as(img)
    y_base = torch.linspace(0, 1, height).repeat(batch_size, width, 1).transpose(1, 2).type_as(img)

    # Apply shift in X direction
    x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)

    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(img, 2 * flow_field - 1, mode="bilinear", padding_mode="zeros", align_corners=False)

    return output


def apply_disparity_v2(img, disp):
    batch_size, _, height, width = img.size()
    disp = disp / (width - 1)

    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, width).repeat(batch_size, height, 1).type_as(img)
    y_base = torch.linspace(0, 1, height).repeat(batch_size, width, 1).transpose(1, 2).type_as(img)

    # Apply shift in X direction
    x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)

    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(img, 2 * flow_field - 1, mode="bilinear", padding_mode="zeros", align_corners=True)

    return output


def compute_reproj_loss_patch(input_L, input_R, pred_disp_l, mask=None, ps=5):
    assert ps % 2 == 1
    bs, c, h, w = input_L.shape
    unfold_func = torch.nn.Unfold(kernel_size=(ps, ps), stride=1, padding=(ps - 1) // 2, dilation=1)
    input_L = unfold_func(input_L)
    input_R = unfold_func(input_R)
    input_L = input_L.reshape(bs, c * ps * ps, h, w)
    input_R = input_R.reshape(bs, c * ps * ps, h, w)
    input_L_warped = apply_disparity_v2(input_R, -pred_disp_l)
    if mask is not None:
        _, new_c, _, _ = input_L.shape
        mask = mask.repeat(1, new_c, 1, 1)
    else:
        mask = torch.ones_like(input_L_warped).type(torch.bool)
    reprojection_loss = F.mse_loss(input_L_warped[mask], input_L[mask])

    return reprojection_loss


def compute_reproj_loss_patch_points(input_L, input_R, points, pred_disp_points, height, width, ps=5):
    assert ps % 2 == 1
    bs, c, h, w = input_L.shape
    unfold_func = torch.nn.Unfold(kernel_size=(ps, ps), stride=1, padding=(ps - 1) // 2, dilation=1)
    input_L = unfold_func(input_L)
    input_R = unfold_func(input_R)
    input_L = input_L.reshape(bs, c * ps * ps, h, w)
    input_R = input_R.reshape(bs, c * ps * ps, h, w)

    u = points[:, 0:1, :] / (width - 1.0)
    v = points[:, 1:2, :] / (height - 1.0)
    pred_disp = pred_disp_points / (width - 1.0)

    uv = torch.cat([u, v], 1).transpose(1, 2)
    uv = uv.unsqueeze(1)  # (B, 1, N, 2)

    uvd = torch.cat([u - pred_disp, v], 1).transpose(1, 2)
    uvd = uvd.unsqueeze(1)

    input_L_points = F.grid_sample(input_L, 2 * uv - 1, mode="bilinear", padding_mode="zeros", align_corners=True)
    input_L_warped_points = F.grid_sample(
        input_R, 2 * uvd - 1, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    reprojection_loss = F.mse_loss(input_L_points, input_L_warped_points)
    return reprojection_loss


def local_contrast_norm(image, kernel_size=9, eps=1e-5):
    """compute local contrast normalization
    input:
        image: torch.tensor (batch_size, 1, height, width)
    output:
        normed_image
    """
    assert kernel_size % 2 == 1, "Kernel size should be odd"
    batch_size, channel, height, width = image.shape
    if channel > 1:
        image = image[:, :1, :, :]
    batch_size, channel, height, width = image.shape
    assert channel == 1, "Only support single channel image for now"
    unfold = nn.Unfold(kernel_size, padding=(kernel_size - 1) // 2)
    unfold_image = unfold(image)  # (batch, kernel_size*kernel_size, height*width)
    avg = torch.mean(unfold_image, dim=1).contiguous().view(batch_size, 1, height, width)
    std = torch.std(unfold_image, dim=1, unbiased=False).contiguous().view(batch_size, 1, height, width)

    normed_image = (image - avg) / (std + eps)
    return normed_image, std


if __name__ == "__main__":
    img_L = torch.rand(1, 1, 256, 512).cuda()
    img_R = torch.rand(1, 1, 256, 512).cuda()
    pred_disp = torch.rand(1, 1, 256, 512).cuda()
    loss, output, mask = compute_reproj_loss_patch(img_L, img_R, pred_disp)
    print(loss)
    print(output.shape)
    print(mask.shape)
