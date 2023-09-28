import os
import random

import cv2
import numpy as np
import torch
from loguru import logger
from path import Path
from PIL import Image
from skimage.feature import canny
from torch.utils.data import Dataset
from tqdm import tqdm

from active_zero2.datasets.data_augmentation import SimIRNoise, data_augmentation
from active_zero2.utils.io import load_pickle
from active_zero2.utils.reprojection import apply_disparity, apply_disparity_v2
from active_zero2.utils.disp_grad import DispGrad


class MessyTableDataset(Dataset):
    def __init__(
        self,
        mode: str,
        domain: str,
        root_dir: str,
        split_file: str,
        height: int,
        width: int,
        meta_name: str,
        depth_name: str,
        normal_name: str,
        normal_conf_name: str,
        left_name: str,
        right_name: str,
        left_pattern_name: str = "",
        right_pattern_name: str = "",
        label_name: str = "",
        num_classes: int = 17,
        depth_r_name: str = "",
        data_aug_cfg=None,
        num_samples: int = 0,
        dilation_factor: int = 10,
        left_off_name="",
        right_off_name="",
        edge_sigma_range=(0.0, 2.0),
    ):
        self.mode = mode

        assert domain in ["sim", "real"], f"Unknown dataset mode: [{domain}]"
        self.domain = domain
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            logger.error(f"Not exists root dir: {self.root_dir}")

        self.split_file = split_file
        self.num_samples = num_samples
        self.dilation_factor = dilation_factor
        self.height, self.width = height, width
        self.meta_name = meta_name
        self.depth_name = depth_name
        self.normal_name = normal_name
        self.normal_conf_name = normal_conf_name
        self.left_name, self.right_name = left_name, right_name
        self.left_pattern_name, self.right_pattern_name = (
            left_pattern_name,
            right_pattern_name,
        )
        self.left_off_name, self.right_off_name = left_off_name, right_off_name
        self.edge_sigma_range = edge_sigma_range
        self.label_name = label_name
        self.num_classes = num_classes
        self.depth_r_name = depth_r_name
        self.sim_ir_noise = SimIRNoise(data_aug_cfg, domain)
        self.data_aug = data_augmentation(data_aug_cfg, domain)

        self.img_dirs = self._gen_path_list()
        if (self.mode == "train" or self.mode == "val") and num_samples:
            self._init_grid()

        self.disp_grad = DispGrad()

        logger.info(
            f"MessyTableDataset: mode: {mode}, domain: {domain}, root_dir: {root_dir}, length: {len(self.img_dirs)},"
            f" left_name: {left_name}, right_name: {right_name},"
            f" left_pattern_name: {left_pattern_name}, right_pattern_name: {right_pattern_name}"
        )

    def _gen_path_list(self):
        img_dirs = []
        if not self.split_file:
            logger.warning(f"Split_file is not defined. The dataset is None.")
            return img_dirs
        with open(self.split_file, "r") as f_split_file:
            for l in f_split_file.readlines():
                img_dirs.append(self.root_dir / l.strip())

        check = False
        if check:
            print("Checking img dirs...")
            for d in tqdm(img_dirs):
                if not d.exists():
                    logger.error(f"{d} not exists.")

        return img_dirs

    def __getitem__(self, index):
        data_dict = {}
        img_dir = self.img_dirs[index]

        img_l = np.array(Image.open(img_dir / self.left_name).convert(mode="L")) / 255  # [H, W]
        img_r = np.array(Image.open(img_dir / self.right_name).convert(mode="L")) / 255

        origin_h, origin_w = img_l.shape[:2]  # # depend on size of your input image, but we hope that is (960, 540)
        origin_h_old, origin_w_old = img_l.shape[:2]  # depend on size of your input image
        if origin_h in (720, 1080):
            img_l = cv2.resize(img_l, (960, 540), interpolation=cv2.INTER_CUBIC)
            img_r = cv2.resize(img_r, (960, 540), interpolation=cv2.INTER_CUBIC)

        origin_h, origin_w = img_l.shape[:2]  # (960, 540)

        assert (
            origin_h == 540 and origin_w == 960
        ), f"Only support H=540, W=960. Current input: H={origin_h}, W={origin_w}"

        if self.left_pattern_name and self.right_pattern_name:
            img_pattern_l = np.array(Image.open(img_dir / self.left_pattern_name).convert(mode="L")) / 255  # [H, W]
            img_pattern_r = np.array(Image.open(img_dir / self.right_pattern_name).convert(mode="L")) / 255
            patter_h, pattern_w = img_pattern_l.shape[:2]
            if patter_h in (720, 1080):
                img_pattern_l = cv2.resize(img_pattern_l, (960, 540), interpolation=cv2.INTER_CUBIC)
                img_pattern_r = cv2.resize(img_pattern_r, (960, 540), interpolation=cv2.INTER_CUBIC)
            patter_h, pattern_w = img_pattern_l.shape[:2]
            assert (
                patter_h == 540 and pattern_w == 960
            ), f"img_pattern_l should be processed to H=540, W=960. {img_dir / self.left_pattern_name}"

        if self.left_off_name and self.right_off_name:
            img_off_l = np.array(Image.open(img_dir / self.left_off_name).convert(mode="L")) / 255  # [H, W]
            img_off_r = np.array(Image.open(img_dir / self.right_off_name).convert(mode="L")) / 255  # [H, W]
            off_h, off_w = img_off_l.shape[:2]
            if off_h in (720, 1080):
                img_off_l = cv2.resize(img_off_l, (960, 540), interpolation=cv2.INTER_CUBIC)
                img_off_r = cv2.resize(img_off_r, (960, 540), interpolation=cv2.INTER_CUBIC)

            off_h, off_w = img_off_l.shape[:2]  # (960, 540)
            assert off_h == 540 and off_w == 960, f"Only support H=540, W=960. Current input: H={off_h}, W={off_w}"

        if self.meta_name:
            img_meta = load_pickle(img_dir / self.meta_name)
            extrinsic_l = img_meta["extrinsic_l"]
            extrinsic_r = img_meta["extrinsic_r"]
            intrinsic_l = img_meta["intrinsic_l"][:3, :3]
            # intrinsic_l[:2] = intrinsic_l[:2] / origin_h_old * 540
            intrinsic_l[:2] = intrinsic_l[:2] / 2
            intrinsic_l[2] = np.array([0.0, 0.0, 1.0])
            baseline = np.linalg.norm(extrinsic_l[:, -1] - extrinsic_r[:, -1])
            focal_length = intrinsic_l[0, 0]
            if self.depth_name:
                img_depth_l = (
                    cv2.imread(img_dir / self.depth_name, cv2.IMREAD_UNCHANGED).astype(float) / 1000
                )  # convert from mm to m
                img_depth_l = cv2.resize(img_depth_l, (origin_w, origin_h), interpolation=cv2.INTER_NEAREST)
                mask = img_depth_l > 0
                img_disp_l = np.zeros_like(img_depth_l)
                img_disp_l[mask] = focal_length * baseline / img_depth_l[mask]
                if self.depth_r_name:
                    img_depth_r = cv2.imread(img_dir / self.depth_r_name, cv2.IMREAD_UNCHANGED).astype(float) / 1000
                    img_depth_r = cv2.resize(img_depth_r, (origin_w, origin_h), interpolation=cv2.INTER_NEAREST)
                    mask = img_depth_r > 0
                    img_disp_r = np.zeros_like(img_depth_r)
                    img_disp_r[mask] = focal_length * baseline / img_depth_r[mask]

        if self.normal_name:
            img_normal_l = cv2.imread(img_dir / self.normal_name, cv2.IMREAD_UNCHANGED)
            img_normal_l = (img_normal_l.astype(float)) / 1000 - 1
            img_normal_l = cv2.resize(img_normal_l, (origin_w, origin_h), interpolation=cv2.INTER_NEAREST)

        if self.normal_conf_name:
            img_normal_weight = cv2.imread(img_dir / self.normal_conf_name, cv2.IMREAD_UNCHANGED)
            img_normal_weight = (img_normal_weight.astype(float)) / 60000.0
            #img_normal_weight *= img_normal_weight
            img_normal_weight = cv2.resize(img_normal_weight, (origin_w, origin_h), interpolation=cv2.INTER_NEAREST)

        if self.label_name:
            img_label_l = cv2.imread(img_dir / self.label_name, cv2.IMREAD_UNCHANGED).astype(int)
            img_label_l = cv2.resize(img_label_l, (origin_w, origin_h), interpolation=cv2.INTER_NEAREST)

        # random crop
        if self.mode == "test":
            x = 0
            y = -2
            assert self.height == 544 and self.width == 960, f"Only support H=544, W=960 for now"

            def crop(img):
                if img.ndim == 2:
                    img = np.concatenate(
                        [np.zeros((2, 960), dtype=img.dtype), img, np.zeros((2, 960), dtype=img.dtype)]
                    )
                else:
                    img = np.concatenate(
                        [
                            np.zeros((2, 960, img.shape[2]), dtype=img.dtype),
                            img,
                            np.zeros((2, 960, img.shape[2]), dtype=img.dtype),
                        ]
                    )
                return img

            def crop_label(img):
                img = np.concatenate(
                    [
                        np.ones((2, 960), dtype=img.dtype) * self.num_classes,
                        img,
                        np.ones((2, 960), dtype=img.dtype) * self.num_classes,
                    ]
                )
                return img

        else:
            x = np.random.randint(0, origin_w - self.width)
            y = np.random.randint(0, origin_h - self.height)

            def crop(img):
                return img[y : y + self.height, x : x + self.width]

            def crop_label(img):
                return img[y : y + self.height, x : x + self.width]

        img_l = crop(img_l)
        img_r = crop(img_r)
        if self.depth_name and self.meta_name:
            intrinsic_l[0, 2] -= x
            intrinsic_l[1, 2] -= y
            img_depth_l = crop(img_depth_l)
            img_disp_l = crop(img_disp_l)
            if self.depth_r_name:
                img_depth_r = crop(img_depth_r)
                img_disp_r = crop(img_disp_r)

        if self.left_pattern_name and self.right_pattern_name:
            img_pattern_l = crop(img_pattern_l)
            img_pattern_r = crop(img_pattern_r)

        if self.left_off_name and self.right_off_name:
            img_off_l = crop(img_off_l)
            img_off_r = crop(img_off_r)

        if self.normal_name:
            img_normal_l = crop(img_normal_l)

        if self.normal_conf_name:
            img_normal_weight = crop(img_normal_weight)

        if self.label_name:
            img_label_l = crop_label(img_label_l)

        data_dict["dir"] = img_dir.name
        data_dict["full_dir"] = str(img_dir)
        img_l = self.sim_ir_noise.apply(img_l)
        img_r = self.sim_ir_noise.apply(img_r)
        data_dict["img_l"] = self.data_aug(img_l).float()
        data_dict["img_r"] = self.data_aug(img_r).float()
        if self.meta_name:
            data_dict["intrinsic_l"] = torch.from_numpy(intrinsic_l).float()
            data_dict["baseline"] = torch.tensor(baseline).float()
            data_dict["focal_length"] = torch.tensor(focal_length).float()
            data_dict["extrinsic_l"] = torch.from_numpy(img_meta["extrinsic_l"])
            data_dict["extrinsic"] = torch.from_numpy(img_meta["extrinsic"])
            data_dict["intrinsic"] = torch.from_numpy(img_meta["intrinsic"])
            if self.depth_name:
                data_dict["img_depth_l"] = torch.from_numpy(img_depth_l).float().unsqueeze(0)
                data_dict["img_disp_l"] = torch.from_numpy(img_disp_l).float().unsqueeze(0)
                data_dict["img_disp_edge_l"] = torch.from_numpy(
                    canny(img_depth_l, sigma=1.0, low_threshold=0.03, high_threshold=0.06)
                ).long()

                if self.depth_r_name:
                    data_dict["img_depth_r"] = torch.from_numpy(img_depth_r).float().unsqueeze(0)
                    data_dict["img_disp_r"] = torch.from_numpy(img_disp_r).float().unsqueeze(0)
                    # disp_grad = self.disp_grad(data_dict["img_disp_r"].unsqueeze(0))[0]
                    data_dict["img_disp_edge_r"] = torch.from_numpy(
                        canny(img_depth_l, sigma=1.0, low_threshold=0.03, high_threshold=0.06)
                    ).long()

                    # compute occlusion
                    img_disp_l_reprojed = apply_disparity_v2(
                        data_dict["img_disp_r"].unsqueeze(0), -data_dict["img_disp_l"].unsqueeze(0)
                    ).squeeze(0)
                    data_dict["img_disp_l_reprojed"] = img_disp_l_reprojed
                    # img_disp_l_reprojed_v2 = apply_disparity_v2(
                    #     data_dict["img_disp_r"].unsqueeze(0), -data_dict["img_disp_l"].unsqueeze(0)
                    # ).squeeze(0)
                    # data_dict["img_disp_l_reprojed_v2"] = img_disp_l_reprojed_v2
                    data_dict["occ_mask_l"] = torch.abs(data_dict["img_disp_l"] - img_disp_l_reprojed) > 1e-1

        if self.left_pattern_name and self.right_pattern_name:
            data_dict["img_pattern_l"] = torch.from_numpy(img_pattern_l).float().unsqueeze(0)
            data_dict["img_pattern_r"] = torch.from_numpy(img_pattern_r).float().unsqueeze(0)

        if self.left_off_name and self.right_off_name:
            if self.mode == "test":
                sigma = (self.edge_sigma_range[0] + self.edge_sigma_range[1]) / 2
            else:
                sigma = (
                    np.random.uniform() * (self.edge_sigma_range[1] - self.edge_sigma_range[0])
                    + self.edge_sigma_range[0]
                )
            img_edge_l = canny(img_off_l, sigma)
            img_edge_r = canny(img_off_r, sigma)
            data_dict["img_edge_l"] = torch.from_numpy(img_edge_l).float().unsqueeze(0)
            data_dict["img_edge_r"] = torch.from_numpy(img_edge_r).float().unsqueeze(0)
            data_dict["img_off_l"] = self.data_aug(img_off_l).float()
            data_dict["img_off_r"] = self.data_aug(img_off_r).float()

        if self.normal_name:
            data_dict["img_normal_l"] = torch.from_numpy(img_normal_l).float().permute(2, 0, 1)
        if self.normal_conf_name:
            data_dict["img_normal_weight"] = torch.from_numpy(img_normal_weight).float()
        if self.label_name:
            data_dict["img_label_l"] = torch.from_numpy(img_label_l).long()
        if (self.mode == "train" or self.mode == "val") and self.num_samples:
            if self.depth_name and self.meta_name:
                sample_data = self._sample_points(img_disp_l)
            else:
                sample_data = self._sample_points()
            data_dict.update(sample_data)
        return data_dict

    def __len__(self):
        return len(self.img_dirs)

    def _init_grid(self):
        nu = np.linspace(0, self.width - 1, self.width)
        nv = np.linspace(0, self.height - 1, self.height)
        u, v = np.meshgrid(nu, nv)

        self.u = u.flatten()
        self.v = v.flatten()

    def _get_coords(self, gt_disp):
        #  Subpixel coordinates
        u = self.u + np.random.random_sample(self.u.size)
        v = self.v + np.random.random_sample(self.v.size)

        u = np.clip(u, 0, self.width - 1)
        v = np.clip(v, 0, self.height - 1)

        # Nearest neighbor
        d = gt_disp[
            np.rint(v).astype(np.uint16),
            np.rint(u).astype(np.uint16),
        ]

        # Remove invalid disparitiy values
        u = u[np.nonzero(d)]
        v = v[np.nonzero(d)]
        d = d[np.nonzero(d)]

        return np.stack((u, v, d), axis=-1)

    def _get_coords_only(self):
        #  Subpixel coordinates
        u = self.u + np.random.random_sample(self.u.size)
        v = self.v + np.random.random_sample(self.v.size)

        u = np.clip(u, 0, self.width - 1)
        v = np.clip(v, 0, self.height - 1)

        return np.stack((u, v), axis=-1)

    # SMDNet
    def _sample_points(self, img_disp_l: np.ndarray or None = None):
        if img_disp_l is None:
            random_points = self._get_coords_only()
            idx = np.random.choice(random_points.shape[0], self.num_samples)
            points = random_points[idx, :]
            return {
                "img_points": torch.from_numpy(points[:, :2].T).float(),
            }
        else:
            edges = get_boundaries(img_disp_l, dilation=self.dilation_factor)
            random_points = self._get_coords(img_disp_l * (1.0 - edges))
            edge_points = self._get_coords(img_disp_l * edges)

            # if edge points exist
            if edge_points.shape[0] > 0:

                # Check tot num of edge points
                cond = edges.sum() // 2 - self.num_samples // 2 < 0
                tot = (
                    (self.num_samples - int(edges.sum()) // 2, int(edges.sum()) // 2)
                    if cond
                    else (self.num_samples // 2, self.num_samples // 2)
                )

                idx = np.random.choice(random_points.shape[0], tot[0])
                idx_edges = np.random.choice(edge_points.shape[0], tot[1])
                points = np.concatenate([edge_points[idx_edges, :], random_points[idx, :]], 0)
            # use uniform sample otherwise
            else:
                random_points = self._get_coords(img_disp_l)
                idx = np.random.choice(random_points.shape[0], self.num_samples)
                points = random_points[idx, :]
            return {
                "img_points": torch.from_numpy(points[:, :2].T).float(),
                "img_labels": torch.from_numpy(points[:, 2:3].T).float(),
            }


# SMDNet
def get_boundaries(disp, th=1.0, dilation=10):
    edges_y = np.logical_or(
        np.pad(np.abs(disp[1:, :] - disp[:-1, :]) > th, ((1, 0), (0, 0))),
        np.pad(np.abs(disp[:-1, :] - disp[1:, :]) > th, ((0, 1), (0, 0))),
    )
    edges_x = np.logical_or(
        np.pad(np.abs(disp[:, 1:] - disp[:, :-1]) > th, ((0, 0), (1, 0))),
        np.pad(np.abs(disp[:, :-1] - disp[:, 1:]) > th, ((0, 0), (0, 1))),
    )
    edges = np.logical_or(edges_y, edges_x).astype(np.float32)

    if dilation > 0:
        kernel = np.ones((dilation, dilation), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

    return edges
