import numpy as np
import argparse
import h5py
from tqdm import tqdm
from path import Path
import cv2
import matplotlib.pyplot as plt
import time
import torch
import os
import open3d as o3d
import sys

_ROOT_DIR = os.path.abspath(os.path.dirname(__file__) + "../../..")
sys.path.insert(0, _ROOT_DIR)
from skimage import feature

from active_zero2.utils.metrics import ErrorMetric
from active_zero2.utils.loguru_logger import setup_logger
from active_zero2.utils.io import load_pickle
from active_zero2.utils.geometry import depth2pts_np


def parse_args():
    parser = argparse.ArgumentParser(description="disp map inpainting using Depth2Depth")
    parser.add_argument("-d", "--data-folder", type=str, required=True)
    parser.add_argument("-p", "--pred-folder", type=str, required=True)
    parser.add_argument(
        "-s",
        "--split-file",
        type=str,
        metavar="FILE",
        required=True,
    )
    parser.add_argument("-c", "--conf-threshold", type=float, required=True)
    parser.add_argument("--cfg", type=str, required=True, help="path to config file")
    parser.add_argument("-n", "--normal-threshold", type=float, required=True)
    parser.add_argument("-a", "--area-threshold", type=float, required=True)
    parser.add_argument("--gt-normal", action="store_true")
    parser.add_argument("--gt-mask", action="store_true")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if args.pred_folder[-1] == "/":
        args.pred_folder = args.pred_folder[:-1]
    data_folder = Path(args.data_folder)
    pred_folder = Path(args.pred_folder)
    if args.gt_normal:
        output_folder = Path(args.pred_folder + "_d2d_gt")
    elif args.gt_mask:
        output_folder = Path(args.pred_folder + "_d2d_gtmask")
    else:
        output_folder = Path(args.pred_folder + "_d2d")

    output_folder.makedirs_p()

    # run name
    timestamp = time.strftime("%y-%m-%d_%H-%M-%S")
    run_name = "{:s}".format(timestamp)
    logger = setup_logger(f"ActiveZero2.test D2D", output_folder, rank=0, filename=f"log.test.{run_name}.txt")
    logger.info(args)

    # evaluate
    from active_zero2.config import cfg
    cfg.merge_from_file(args.cfg)
    cfg.TEST.MAX_DISP = 128

    metric = ErrorMetric(
        model_type="Depth2Depth",
        use_mask=cfg.TEST.USE_MASK,
        max_disp=cfg.TEST.MAX_DISP,
        depth_range=cfg.TEST.DEPTH_RANGE,
        num_classes=cfg.DATA.NUM_CLASSES,
        is_depth=True,
    )
    metric.reset()

    from active_zero2.datasets.messytable import MessyTableDataset

    dataset = MessyTableDataset(
        mode="test",
        domain="real",
        root_dir=args.data_folder,
        split_file=args.split_file,
        height=544,
        width=960,
        meta_name="meta.pkl",
        depth_name="depthL.png",
        normal_name="normalL.png",
        normal_conf_name="",
        left_name="1024_irL_real_360.png",
        right_name="1024_irR_real_360.png",
        left_pattern_name="",
        right_pattern_name="",
        label_name="labelL.png",
    )

    TRANSPARENT_LABELS = [9, 13, 16]

    eval_tic = time.time()
    for data in tqdm(dataset):
        sc = data["dir"]
        print(sc)
        view_folder = output_folder / sc
        view_folder.makedirs_p()
        curr_pred_dir = pred_folder / f"{sc}_real"

        depth_pred = np.load(curr_pred_dir / "depthL_pred.npy")
        depth_pred_bak = depth_pred.copy()
        depth_mask = (depth_pred > cfg.TEST.DEPTH_RANGE[0]) * (depth_pred < cfg.TEST.DEPTH_RANGE[1])
        depth_pred = depth_pred * depth_mask

        meta = load_pickle(data_folder / sc / "meta.pkl")
        d2d_intrinsic = meta["intrinsic_l"][:3,:3].copy()
        d2d_intrinsic[:2] /= 2
        if args.gt_mask:
            label_l = data["img_label_l"].cpu().numpy()[2:-2]
            mask = np.logical_or.reduce([label_l == x for x in TRANSPARENT_LABELS])
            mask = np.logical_not(mask)
            plt.imsave(view_folder / "obj_mask.png", mask.astype(float), vmin=0.0, vmax=1.0, cmap="jet")
            depth_pred = depth_pred * mask
            normal_pred = cv2.imread(curr_pred_dir / "normal_pred3_u16.png", cv2.IMREAD_UNCHANGED)
            normal_pred = (normal_pred.astype(float)) / 1000.0 - 1
            normal_pred2 = np.transpose(normal_pred, (2, 0, 1))
            normal_pred2 = np.stack([normal_pred2[0], normal_pred2[2], -normal_pred2[1]])
            with h5py.File(view_folder / "pred_normal.h5", "w") as f:
                f.create_dataset("result", data=normal_pred2)
        else:
            # left image
            conf_map = np.load(curr_pred_dir / "confidence.npy")
            conf_mask = conf_map[3] > args.conf_threshold
            conf_mask_bak = conf_mask.copy()
            plt.imsave(view_folder / "conf_mask1.png", conf_mask.astype(float), vmin=0.0, vmax=1.0, cmap="jet")
            conf_map = np.clip((conf_map[3] - args.conf_threshold) / (1 - args.conf_threshold), 0, 1)
            cv2.imwrite(view_folder / "conf_map_u16.png", (conf_map * 65535).astype(np.uint16))
            conf_mask = conf_mask.astype(np.uint8)
            # remove unconnected small regions
            ret, comps, stats, _ = cv2.connectedComponentsWithStats(conf_mask)
            inv_idx = np.where(stats[:, -1] < args.area_threshold)[0]
            inv_mask = np.logical_or.reduce([comps == x for x in inv_idx])
            conf_mask *= np.logical_not(inv_mask)
            plt.imsave(view_folder / "conf_mask2.png", conf_mask.astype(float), vmin=0.0, vmax=1.0, cmap="jet")

            normal_fitting = cv2.imread(curr_pred_dir / "normalL_fitting_u16.png", cv2.IMREAD_UNCHANGED)
            normal_fitting = (normal_fitting.astype(float)) / 1000.0 - 1

            if args.gt_normal:
                normal_gt = cv2.imread(data_folder / sc / "normalL.png", cv2.IMREAD_UNCHANGED)
                normal_gt = (normal_gt.astype(float)) / 1000.0 - 1
                normal_gt = cv2.resize(normal_gt, (960, 540), interpolation=cv2.INTER_NEAREST)
                normal_diff = np.arccos(np.clip(np.sum(normal_fitting * normal_gt, axis=-1), -1, 1))
                normal_mask = normal_diff < args.normal_threshold / 180 * np.pi
                cv2.imwrite(view_folder / "normal_mask.png", (normal_mask.astype(np.uint8)) * 255)
                # normal_gt = cv2.resize(normal_gt, (480, 270), interpolation=cv2.INTER_NEAREST)
                normal_gt2 = np.transpose(normal_gt, (2, 0, 1))
                normal_gt2 = np.stack([normal_gt2[0], normal_gt2[2], -normal_gt2[1]])
                with h5py.File(view_folder / "gt_normal.h5", "w") as f:
                    f.create_dataset("result", data=normal_gt2)
            else:
                normal_pred = cv2.imread(curr_pred_dir / "normalL_pred3_u16.png", cv2.IMREAD_UNCHANGED)
                normal_pred = (normal_pred.astype(float)) / 1000.0 - 1
                normal_diff = np.arccos(np.clip(np.sum(normal_fitting * normal_pred, axis=-1), -1, 1))
                normal_mask = normal_diff < args.normal_threshold / 180 * np.pi
                cv2.imwrite(view_folder / "normal_mask.png", (normal_mask.astype(np.uint8)) * 255)
                normal_pred2 = np.transpose(normal_pred, (2, 0, 1))
                normal_pred2 = np.stack([normal_pred2[0], normal_pred2[2], -normal_pred2[1]])
                with h5py.File(view_folder / "pred_normal.h5", "w") as f:
                    f.create_dataset("result", data=normal_pred2)
            depth_pred = depth_pred * normal_mask
            depth_pred = depth_pred * conf_mask

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(depth2pts_np(depth_pred, d2d_intrinsic))
        pcd = pcd.crop(
            o3d.geometry.AxisAlignedBoundingBox(min_bound=np.array([-10, -10, 0.1]), max_bound=np.array([10, 10, 1.8]))
        )
        o3d.io.write_point_cloud(view_folder / "d2d_in.pcd", pcd)
        cv2.imwrite(view_folder / "depthL_pred_u16.png", (depth_pred * 4000).astype(np.uint16))
        depthL_pred_colored = depth_pred
        vis_depth = visualize_depth(depthL_pred_colored)
        cv2.imwrite(os.path.join(view_folder, "depth_pred_colored.png"), vis_depth)

        edge = cv2.imread(curr_pred_dir / "edge_pred.png", 0)
        kernel = np.ones((3, 3), dtype=np.uint8) * 255
        edge = cv2.dilate(edge, kernel, iterations=3)
        edge = edge.astype(np.uint16)
        cv2.imwrite(view_folder / "edge_pred_u16.png", (1021 - edge * 4))

        cpp_tic = time.time()
        print("Execute Depth2Depth...")
        o = (
            f"/home/rvsa/Downloads/cleargrasp-master/api/depth2depth/gaps/bin/x86_64/depth2depth"
            f" {view_folder}/depthL_pred_u16.png {view_folder}/depthL_pred_u16_out.png -xres 960 -yres 540"
            f" -input_tangent_weight {view_folder}/edge_pred_u16.png"
            f" -fx {meta['intrinsic_l'][0,0]/2} -fy {meta['intrinsic_l'][1,1]/2}"
            f" -cx {meta['intrinsic_l'][0,2]/2} -cy {meta['intrinsic_l'][1,2]/2}"
            f" -inertia_weight 1000 -smoothness_weight 0.001 -tangent_weight 10"
        )
        if args.gt_normal:
            o += f" -input_normals {view_folder}/gt_normal.h5"
        else:
            o += f" -input_normals {view_folder}/pred_normal.h5"
        print(o)
        os.system(o)
        cpp_time = time.time() - cpp_tic
        logger.info(f"{sc} Depth2Depth time: {cpp_time:.2f}s")

        depth_d2d = cv2.imread(view_folder / "depthL_pred_u16_out.png", cv2.IMREAD_UNCHANGED)
        depth_d2d = (depth_d2d.astype(float)) / 4000.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(depth2pts_np(depth_d2d, d2d_intrinsic))
        pcd = pcd.crop(
            o3d.geometry.AxisAlignedBoundingBox(min_bound=np.array([-10, -10, 0.1]), max_bound=np.array([10, 10, 1.8]))
        )
        o3d.io.write_point_cloud(view_folder / "d2d.pcd", pcd)

        depth_d2d = cv2.resize(depth_d2d, (960, 540), interpolation=cv2.INTER_LANCZOS4)

        depth_d2d_mask = (depth_d2d > cfg.TEST.DEPTH_RANGE[0]) * (depth_d2d < cfg.TEST.DEPTH_RANGE[1])
        plt.imsave(view_folder / "depth_d2d_mask.png", depth_d2d_mask.astype(float), vmin=0.0, vmax=1.0, cmap="jet")
        if args.gt_mask:
            depth_d2d_mask = np.logical_and(depth_d2d_mask, np.logical_not(mask))
            depth_d2d_fusion = depth_d2d * depth_d2d_mask + depth_pred_bak * (1 - depth_d2d_mask)
        else:
            depth_d2d_mask = np.logical_and(depth_d2d_mask, np.logical_not(conf_mask_bak))
            # depth_d2d_mask = np.logical_and(depth_d2d_mask, np.logical_not(normal_mask))
            depth_d2d_fusion = depth_d2d * depth_d2d_mask + depth_pred_bak * (1 - depth_d2d_mask)
        np.save(view_folder / "depth_d2d_fusion.npy", depth_d2d_fusion)

        pred_dict = {"depth": depth_d2d_fusion}
        data = {k: v.unsqueeze(0) for k, v in data.items() if isinstance(v, torch.Tensor)}
        data["dir"] = sc
        curr_metric = metric.compute(data, pred_dict, save_folder=view_folder, real_data=True)
        logger.info("\t".join([f"{k}: {v:.3f}" for k, v in curr_metric.items()]))

    # END
    epoch_time_eval = time.time() - eval_tic
    logger.info("Real Test total_time: {:.2f}s".format(epoch_time_eval))
    logger.info("Real Eval Metric: \n" + metric.summary())
    logger.info("\nConfident threshold:" + str(args.conf_threshold))
    logger.info("\nNormal threshold:" + str(args.normal_threshold))

def visualize_depth(depth):
    MAX_DEPTH = 2.0
    cmap = plt.get_cmap("rainbow")
    if depth.dtype == np.uint16:
        depth = depth.astype(np.float32) / 1000.0
    if len(depth.shape) == 3:
        depth = depth[..., 0]
    depth = np.clip(depth / MAX_DEPTH, 0.0, 1.0)
    vis_depth = cmap(depth)
    vis_depth = (vis_depth[:, :, :3] * 255.0).astype(np.uint8)
    vis_depth = cv2.cvtColor(vis_depth, cv2.COLOR_RGB2BGR)
    return vis_depth

if __name__ == "__main__":
    main()
