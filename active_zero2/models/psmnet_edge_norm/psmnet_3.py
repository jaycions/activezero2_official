import math

import torch

from active_zero2.models.psmnet_edge_norm.psmnet_submodule_3 import *
from active_zero2.utils.confidence import compute_confidence
from active_zero2.utils.reprojection import compute_reproj_loss_patch
from active_zero2.utils.disp_grad import DispGrad


class hourglass(nn.Module):
    def __init__(self, inplanes, dilation):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, inplanes * 2, kernel_size=3, stride=2, padding=dilation, dilation=dilation),
            nn.BatchNorm3d(inplanes * 2),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, padding=dilation, dilation=dilation),
            nn.BatchNorm3d(inplanes * 2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, padding=dilation, dilation=dilation),
            nn.BatchNorm3d(inplanes * 2),
            nn.ReLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, padding=dilation, dilation=dilation),
            nn.BatchNorm3d(inplanes * 2, inplanes * 2),
            nn.ReLU(inplace=True),
        )

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(
                inplanes * 2,
                inplanes * 2,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
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
                padding=dilation,
                dilation=dilation,
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


class PSMNetEdgeNormal(nn.Module):
    def __init__(
        self,
        min_disp: float,
        max_disp: float,
        num_disp: int,
        set_zero: bool,
        dilation: int,
        epsilon: float,
        grad_threshold: float,
        use_off: bool,
        use_volume: bool,
        edge_weight: float,
    ):
        super(PSMNetEdgeNormal, self).__init__()
        self.min_disp = min_disp
        self.max_disp = max_disp
        self.num_disp = num_disp
        self.dilation = dilation
        self.epsilon = epsilon
        self.use_off = use_off
        self.use_volume = use_volume
        self.edge_weight = edge_weight
        assert num_disp % 4 == 0, "Num_disp % 4 should be 0"
        self.num_disp_4 = num_disp // 4
        self.set_zero = set_zero  # set zero for invalid reference image cost volume

        self.disp_list = torch.linspace(min_disp, max_disp, num_disp)
        self.disp_list_4 = torch.linspace(min_disp, max_disp, self.num_disp_4) / 4
        self.disp_regression = DisparityRegression(min_disp, max_disp, num_disp)

        self.feature_extraction = FeatureExtraction(1)

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

        self.dres2 = hourglass(32, dilation)
        self.dres3 = hourglass(32, dilation)
        self.dres4 = hourglass(32, dilation)

        self.dres0n = nn.Sequential(
            convbn_3d(64, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            convbn_3d(16, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.dres1n = nn.Sequential(
            convbn_3d(16, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            convbn_3d(16, 16, 3, 1, 1),
        )
        self.dres2n = hourglass(16, dilation)
        self.dres3n = hourglass(16, dilation)
        self.dres4n = hourglass(16, dilation)

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
        self.normal1 = nn.Sequential(
            convbn_3d(16, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 3, kernel_size=3, padding=1, stride=1, bias=False),
        )
        self.normal2 = nn.Sequential(
            convbn_3d(16, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 3, kernel_size=3, padding=1, stride=1, bias=False),
        )
        self.normal3 = nn.Sequential(
            convbn_3d(16, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 3, kernel_size=3, padding=1, stride=1, bias=False),
        )

        self.img_ir_conv = nn.Sequential(
            convbn(1, 32, 3, 2, 1, 1),
            nn.ReLU(inplace=True),
            convbn(32, 32, 3, 2, 1, 1),
            nn.ReLU(inplace=True),
            convbn(32, 32, 3, 1, 1, 1),
            nn.ReLU(inplace=True),
        )

        if self.use_off:
            self.img_off_conv = nn.Sequential(
                convbn(1, 32, 3, 2, 1, 1),
                nn.ReLU(inplace=True),
                convbn(32, 32, 3, 2, 1, 1),
                nn.ReLU(inplace=True),
                convbn(32, 32, 3, 1, 1, 1),
                nn.ReLU(inplace=True),
            )

        in_channels = 32 + 32
        if self.use_off:
            in_channels += 32
        if self.use_volume:
            in_channels += self.num_disp_4
        self.edge_conv = nn.Sequential(
            convbn(in_channels, 32, 3, 1, 1, 1),
            nn.ReLU(inplace=True),
            convbn(32, 32, 3, 1, 1, 1),
            nn.ReLU(inplace=True),
            convbn(32, 32, 3, 1, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 1, bias=False),
        )

        self.edge_loss = nn.CrossEntropyLoss(weight=torch.Tensor([1, edge_weight]), reduction="none")

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

        self.disp_grad = DispGrad(grad_threshold)

    def forward(self, data_batch):
        img_L, img_R = data_batch["img_l"], data_batch["img_r"]
        refimg_feature = self.feature_extraction(img_L)  # [bs, 32, H/4, W/4]
        targetimg_feature = self.feature_extraction(img_R)

        disp_list = self.disp_list.to(refimg_feature.device)
        disp_list_4 = self.disp_list_4.to(refimg_feature.device)

        # Cost Volume
        [bs, feature_size, H, W] = refimg_feature.size()
        # Original coordinates of pixels
        x_base = (
            torch.linspace(0, 1, W, dtype=refimg_feature.dtype, device=refimg_feature.device)
            .view(1, 1, W, 1)
            .expand(bs, H, W, self.num_disp_4)
        )
        y_base = (
            torch.linspace(0, 1, H, dtype=refimg_feature.dtype, device=refimg_feature.device)
            .view(1, H, 1, 1)
            .expand(bs, H, W, self.num_disp_4)
        )
        disp_grid = (disp_list_4 / (W - 1)).view(1, 1, 1, self.num_disp_4).expand(bs, H, W, self.num_disp_4)
        target_grids = torch.stack((x_base - disp_grid, y_base), dim=-1).view(bs, H, W * self.num_disp_4, 2)
        target_cost_volume = F.grid_sample(
            targetimg_feature, 2 * target_grids - 1, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        target_cost_volume = target_cost_volume.view(bs, feature_size, H, W, self.num_disp_4).permute(0, 1, 4, 2, 3)
        ref_cost_volume = refimg_feature.unsqueeze(2).expand(bs, feature_size, self.num_disp_4, H, W)
        if self.set_zero:
            # set invalid area to zero
            valid_mask = (x_base > disp_grid).permute(0, 3, 1, 2).unsqueeze(1)
            ref_cost_volume = ref_cost_volume * valid_mask

        cost = torch.cat((ref_cost_volume, target_cost_volume), dim=1)

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

        cost0n = self.dres0n(cost)
        cost0n = self.dres1n(cost0n) + cost0n
        out1n, pre1n, post1n = self.dres2n(cost0n, None, None)
        out1n = out1n + cost0n

        out2n, pre2n, post2n = self.dres3n(out1n, pre1n, post1n)
        out2n = out2n + cost0n

        out3n, pre3n, post3n = self.dres4n(out2n, pre1n, post2n)
        out3n = out3n + cost0n

        normal1 = self.normal1(out1n)
        normal2 = self.normal2(out2n) + normal1
        normal3 = self.normal3(out3n) + normal2

        if self.training:
            cost1 = F.interpolate(
                cost1,
                (self.num_disp, 4 * H, 4 * W),
                mode="trilinear",
                align_corners=False,
            )
            cost2 = F.interpolate(
                cost2,
                (self.num_disp, 4 * H, 4 * W),
                mode="trilinear",
                align_corners=False,
            )

            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = self.disp_regression(pred1)

            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = self.disp_regression(pred2)

        prob_volume = F.softmax(torch.squeeze(cost3, 1), 1)
        cost3 = F.interpolate(cost3, (self.num_disp, 4 * H, 4 * W), mode="trilinear", align_corners=False)
        cost3 = torch.squeeze(cost3, 1)
        prob_cost3 = F.softmax(cost3, dim=1)

        pred3 = self.disp_regression(prob_cost3)

        normal3 = F.interpolate(normal3, (self.num_disp_4, 4 * H, 4 * W), mode="trilinear", align_corners=False)
        normal3 = torch.sum(normal3, dim=2)
        normal3 = normal3 / torch.linalg.norm(normal3, dim=1, keepdim=True)

        # edge prediction
        inputs = []
        img_feature = self.img_ir_conv(img_L)
        inputs.append(img_feature)
        inputs.append(refimg_feature)
        if self.use_off:
            img_off_feature = self.img_off_conv(data_batch["img_off_l"])
            inputs.append(img_off_feature)
        if self.use_volume:
            inputs.append(prob_volume)
        inputs = torch.cat(inputs, dim=1)
        inputs = F.interpolate(inputs, (4 * H, 4 * W), mode="bilinear", align_corners=False)
        edge = self.edge_conv(inputs)

        if self.training:
            pred_dict = {
                "pred1": pred1,
                "pred2": pred2,
                "pred3": pred3,
                "edge": edge,
                "normal3": normal3,
            }
        else:
            pred_norm = (pred3 - self.min_disp) / (self.max_disp - self.min_disp)
            conf_map = compute_confidence(pred_norm, prob_cost3)
            pred_dict = {
                "pred3": pred3,
                "prob_volume": prob_volume,
                "conf_map": conf_map,
                "edge": edge,
                "normal3": normal3,
            }

        return pred_dict

    def compute_disp_loss(self, data_batch, pred_dict):
        disp_gt = data_batch["img_disp_l"]
        # Get stereo loss on sim
        # Note in training we do not exclude bg
        mask = (disp_gt < self.max_disp) * (disp_gt > self.min_disp)
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
            mask = (disp_gt < self.max_disp) * (disp_gt > self.min_disp)
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

    def compute_edge_loss(
        self,
        data_batch,
        pred_dict,
    ):
        disp_gt = data_batch["img_disp_l"]
        mask = (disp_gt < self.max_disp) * (disp_gt > self.min_disp)
        mask.detach()
        edge_pred = pred_dict["edge"]
        edge_gt = data_batch["img_disp_edge_l"]
        edge_loss = self.edge_loss(edge_pred, edge_gt)
        edge_loss = (edge_loss * mask).sum() / (mask.sum() + 1e-7)

        with torch.no_grad():
            mask = mask.squeeze(1)
            edge_pred = torch.argmax(edge_pred, dim=1)
            tp = (edge_pred == 1) * (edge_gt == 1) * mask
            prec = tp.sum() / (((edge_pred == 1) * mask).sum() + 1e-7)
            recall = tp.sum() / ((edge_gt[mask] == 1).sum() + 1e-7)

        return edge_loss, prec, recall

    def compute_grad_loss(self, data_batch, pred_dict):
        disp_pred = pred_dict["pred3"]
        disp_grad_pred = self.disp_grad(disp_pred)

        if "img_disp_l" in data_batch:
            disp_gt = data_batch["img_disp_l"]
            disp_grad_gt = self.disp_grad(disp_gt)
            grad_diff = torch.abs(disp_grad_pred - disp_grad_gt)
            grad_diff = grad_diff * torch.exp(-torch.abs(disp_grad_gt) * self.epsilon)
            mask = (disp_gt < self.max_disp) * (disp_gt > self.min_disp)
            mask.detach()
            loss = torch.mean(grad_diff * mask)
        else:
            loss = torch.mean(torch.abs(disp_grad_pred))
        return loss

    def compute_normal_loss(self, data_batch, pred_dict):
        normal_gt = data_batch["img_normal_l"]
        normal_pred = pred_dict["normal3"]
        cos = F.cosine_similarity(normal_gt, normal_pred, dim=1, eps=1e-6)
        loss_cos = 1.0 - cos
        if "img_disp_l" in data_batch:
            disp_gt = data_batch["img_disp_l"]
            mask = (disp_gt < self.max_disp) * (disp_gt > self.min_disp)
            mask = mask.squeeze(1)
            mask.detach()
        else:
            mask = torch.ones_like(loss_cos)

        if "img_normal_weight" in data_batch:
            img_normal_weight = data_batch["img_normal_weight"]  # (B, H, W)
            loss_cos = (loss_cos * img_normal_weight * mask).sum() / (img_normal_weight * mask).sum()
        else:
            loss_cos = (loss_cos * mask).sum() / mask.sum()
        return loss_cos


if __name__ == "__main__":
    model = PSMNetDilation(min_disp=12, max_disp=96, num_disp=128, set_zero=False, dilation=3)
    model = model.cuda()
    model.eval()

    data_batch = {
        "img_l": torch.rand(1, 1, 256, 512).cuda(),
        "img_r": torch.rand(1, 1, 256, 512).cuda(),
    }
    pred = model(data_batch)

    for k, v in pred.items():
        print(k, v.shape)
