"""
Author: Isabella Liu 11/10/21
Feature: Hourglass and PSMNet (stacked hourglass) module without small transform layers
"""

import math

from active_zero2.models.psmnet.psmnet_submodule_3 import *
from active_zero2.utils.reprojection import compute_reproj_loss_patch


class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(
            convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1),
            nn.ReLU(inplace=True),
        )

        self.conv2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(
            convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1),
            nn.ReLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1),
            nn.ReLU(inplace=True),
        )

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(
                inplanes * 2,
                inplanes * 2,
                kernel_size=3,
                padding=1,
                output_padding=1,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm3d(inplanes * 2),
        )

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(
                inplanes * 2,
                inplanes,
                kernel_size=3,
                padding=1,
                output_padding=1,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm3d(inplanes),
        )

    def forward(self, x, presqu, postqu):
        out = self.conv1(x)
        pre = self.conv2(out)
        if postqu is not None:
            pre = F.relu(pre + postqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)
        out = self.conv4(out)

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)
        return out, pre, post


class PSMNet(nn.Module):
    def __init__(self, maxdisp):
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp

        self.init_dim = 32 + self.maxdisp // 4

        self.feature_extraction = FeatureExtraction()

        self.dres0 = nn.Sequential(
            convbn_3d(64, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.dres1 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            convbn_3d(32, 32, 3, 1, 1),
        )

        self.dres2 = hourglass(32)
        self.dres3 = hourglass(32)
        self.dres4 = hourglass(32)

        self.classif1 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False),
        )
        self.classif2 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False),
        )
        self.classif3 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, data_batch):
        img_L, img_R = data_batch["img_l"], data_batch["img_r"]
        refimg_feature = self.feature_extraction(img_L)  # [bs, 32, H/4, W/4]
        targetimg_feature = self.feature_extraction(img_R)

        # Cost Volume
        [bs, feature_size, H, W] = refimg_feature.size()
        cost = torch.FloatTensor(bs, feature_size * 2, self.maxdisp // 4, H, W).zero_().to(refimg_feature.device)

        for i in range(self.maxdisp // 4):
            if i > 0:
                cost[:, :feature_size, i, :, i:] = refimg_feature[:, :, :, i:]
                cost[:, feature_size:, i, :, i:] = targetimg_feature[:, :, :, :-i]
            else:
                cost[:, :feature_size, i, :, :] = refimg_feature
                cost[:, feature_size:, i, :, :] = targetimg_feature
        cost = cost.contiguous()  # [bs, fs*2, max_disp/4, H/4, W/4]

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1 + cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2 + cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3 + cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2
        smd_cost = cost3

        if self.training:
            cost1 = F.interpolate(
                cost1,
                (self.maxdisp, 4 * H, 4 * W),
                mode="trilinear",
                align_corners=False,
            )
            cost2 = F.interpolate(
                cost2,
                (self.maxdisp, 4 * H, 4 * W),
                mode="trilinear",
                align_corners=False,
            )

            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = DisparityRegression(self.maxdisp)(pred1)

            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = DisparityRegression(self.maxdisp)(pred2)

        cost3 = F.interpolate(cost3, (self.maxdisp, 4 * H, 4 * W), mode="trilinear", align_corners=False)
        cost3 = torch.squeeze(cost3, 1)
        pred3 = F.softmax(cost3, dim=1)

        # For your information: This formulation 'softmax(c)' learned "similarity"
        # while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
        # However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
        pred3 = DisparityRegression(self.maxdisp)(pred3)
        if self.training:
            pred_dict = {
                "pred1": pred1,
                "pred2": pred2,
                "pred3": pred3,
                "cost3": torch.squeeze(smd_cost, 1),
                "refimg_fea": refimg_feature,
            }
        else:
            pred_dict = {
                "pred3": pred3,
                "cost3": torch.squeeze(smd_cost, 1),
                "refimg_fea": refimg_feature,
            }

        return pred_dict

    def compute_disp_loss(self, data_batch, pred_dict):
        disp_gt = data_batch["img_disp_l"]
        # Get stereo loss on sim
        # Note in training we do not exclude bg
        mask = (disp_gt < self.maxdisp) * (disp_gt > 0)
        mask.detach()
        loss_disp = 0.0
        for pred_name, loss_weight in zip(["pred1", "pred2", "pred3"], [0.5, 0.7, 1.0]):
            if pred_name in pred_dict:
                loss_disp += loss_weight * F.smooth_l1_loss(pred_dict[pred_name][mask], disp_gt[mask], reduction="mean")

        return loss_disp

    def compute_reproj_loss(self, data_batch, pred_dict, use_mask: bool, patch_size: int, only_last_pred: bool):
        if use_mask:
            disp_gt = data_batch["img_disp_l"]
            # Get stereo loss on sim
            # Note in training we do not exclude bg
            mask = (disp_gt < self.maxdisp) * (disp_gt > 0)
            mask.detach()
        else:
            mask = None
        if only_last_pred:
            loss_reproj = compute_reproj_loss_patch(
                data_batch["img_pattern_l"],
                data_batch["img_pattern_r"],
                pred_disp_l=pred_dict["pred3"],
                mask=mask,
                ps=patch_size,
            )

            return loss_reproj
        else:
            loss_reproj = 0.0
            for pred_name, loss_weight in zip(["pred1", "pred2", "pred3"], [0.5, 0.7, 1.0]):
                if pred_name in pred_dict:
                    loss_reproj += loss_weight * compute_reproj_loss_patch(
                        data_batch["img_pattern_l"],
                        data_batch["img_pattern_r"],
                        pred_disp_l=pred_dict[pred_name],
                        mask=mask,
                        ps=patch_size,
                    )
            return loss_reproj


if __name__ == "__main__":
    # Test PSMNet
    model = PSMNet(maxdisp=192).cuda()
    model.eval()
    img_L = torch.rand(1, 3, 256, 512).cuda()
    img_R = torch.rand(1, 3, 256, 512).cuda()
    pred = model(img_L, img_R)
    print(f"pred shape {pred.shape}")  # pred shape torch.Size([1, 1, 256, 512])
