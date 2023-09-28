import argparse
import copy
import sys
import os
import os.path as osp

_ROOT_DIR = os.path.abspath(osp.join(osp.dirname(__file__), ".."))
sys.path.insert(0, _ROOT_DIR)
import time
from path import Path
import numpy as np
import cv2
import csv
from tabulate import tabulate
from tqdm import tqdm

from active_zero2.utils.loguru_logger import setup_logger
import matplotlib.pyplot as plt

with open(osp.join(_ROOT_DIR, "data_rendering/materials/objects.csv"), "r") as f:
    OBJECT_INFO = csv.reader(f)
    OBJECT_INFO = list(OBJECT_INFO)[1:]
    OBJECT_NAMES = [_[0] for _ in OBJECT_INFO]

REAL_OBJECTS = [
    "coca_cola",
    "coffee_cup",
    "gold_ball",
    "jack_daniels",
    "spellegrino",
    "steel_ball",
    "tennis_ball",
    "voss",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate real realsense performance")
    parser.add_argument("-d", "--data-folder", type=str, required=True)
    parser.add_argument("-p", "--pred-folder", type=str, required=True)
    parser.add_argument(
        "-s",
        "--split-file",
        type=str,
        metavar="FILE",
        required=True,
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if args.pred_folder[-1] == "/":
        args.pred_folder = args.pred_folder[:-1]
    data_folder = Path(args.data_folder)
    pred_folder = Path(args.pred_folder)

    timestamp = time.strftime("%y-%m-%d_%H-%M-%S")
    run_name = "{:s}".format(timestamp)
    logger = setup_logger(f"ActiveZero2.test Normal3", pred_folder, rank=0, filename=f"log.normal3.{run_name}.txt")
    logger.info(args)

    with open(args.split_file, "r") as f:
        scene_list = [line.strip() for line in f]

    num_classes = 17
    normal_err = []
    normal_err10 = []
    normal_err20 = []
    obj_normal_err = np.zeros(num_classes)
    obj_normal_err10 = np.zeros(num_classes)
    obj_count = np.zeros(num_classes)
    real_normal_err = 0.0
    real_normal_err10 = 0.0
    real_count = 0
    print_normal_err = 0.0
    print_normal_err10 = 0.0
    print_count = 0

    for scene in tqdm(scene_list):
        scene_data = data_folder / scene
        scene_pred = pred_folder / scene + "_real"

        normal_gt = cv2.imread(scene_data / "normalL.png", cv2.IMREAD_UNCHANGED)
        normal_gt = (normal_gt.astype(float) / 1000.0) - 1
        normal_gt = cv2.resize(normal_gt, (960, 540), interpolation=cv2.INTER_NEAREST)
        label_l = cv2.imread(scene_data / "irL_label_image.png", cv2.IMREAD_UNCHANGED).astype(int)
        label_l = cv2.resize(label_l, (960, 540), interpolation=cv2.INTER_NEAREST)

        normal_pred = cv2.imread(scene_pred / "normal_pred3_u16.png", cv2.IMREAD_UNCHANGED)
        normal_pred = (normal_pred.astype(float) / 1000.0) - 1

        mask = cv2.imread(scene_pred / "mask.png")
        valid_mask = mask[..., 2] > 100
        invalid_mask = np.logical_not(valid_mask)

        normal_diff = np.arccos(np.clip(np.sum(normal_gt * normal_pred, axis=-1), -1, 1))
        plt.imsave(
            os.path.join(scene_pred, "normal_diffE.png"),
            normal_diff * valid_mask,
            vmin=0.0,
            vmax=np.pi/2,
            cmap="jet",
        )
        normal_diff[invalid_mask] = 0
        normal_err.append(normal_diff.sum() / valid_mask.sum())
        normal_err10.append((normal_diff > 10 / 180 * np.pi).sum() / valid_mask.sum())
        normal_err20.append((normal_diff > 20 / 180 * np.pi).sum() / valid_mask.sum())

        for i in range(num_classes):
            obj_mask = label_l == i
            obj_mask = np.logical_and(obj_mask, valid_mask)
            if obj_mask.sum() > 0:
                obj_count[i] += 1
                obj_normal_err[i] += (normal_diff[obj_mask]).mean()
                obj_normal_err10[i] += (normal_diff[obj_mask] > 10 / 180 * np.pi).sum() / obj_mask.sum()
                if OBJECT_NAMES[i] in REAL_OBJECTS:
                    real_count += 1
                    real_normal_err += (normal_diff[obj_mask]).mean()
                    real_normal_err10 += (normal_diff[obj_mask] > 10 / 180 * np.pi).sum() / obj_mask.sum()
                else:
                    print_count += 1
                    print_normal_err += (normal_diff[obj_mask]).mean()
                    print_normal_err10 += (normal_diff[obj_mask] > 10 / 180 * np.pi).sum() / obj_mask.sum()

    s = "Normal evaluation result: \n"
    table = [[]]
    headers = ["norm_err", "norm_err10", "norm_err20"]
    table[0] = [np.mean(normal_err), np.mean(normal_err10), np.mean(normal_err20)]
    s += tabulate(table, headers=headers, tablefmt="fancy_grid", floatfmt=".4f")

    if obj_count.sum() > 0:
        headers = ["class_id", "name", "count", "obj_norm_err", "obj_norm_err10"]
        table = []
        for i in range(num_classes):
            t = [
                i,
                OBJECT_NAMES[i],
                obj_count[i],
            ]
            t += [
                obj_normal_err[i] / (obj_count[i] + 1e-7),
                obj_normal_err10[i] / (obj_count[i] + 1e-7),
            ]
            table.append(t)
        t = [
            "-",
            "REAL",
            real_count,
        ]
        t += [
            real_normal_err / (real_count + 1e-7),
            real_normal_err10 / (real_count + 1e-7),
        ]
        table.append(t)
        t = [
            "-",
            "PRINT",
            print_count,
        ]
        t += [
            print_normal_err / (print_count + 1e-7),
            print_normal_err10 / (print_count + 1e-7),
        ]
        table.append(t)
        t = [
            "-",
            "ALL",
            print_count + real_count,
        ]
        t += [
            (print_normal_err + real_normal_err) / (print_count + real_count + 1e-7),
            (print_normal_err10 + real_normal_err10) / (print_count + real_count + 1e-7),
        ]
        table.append(t)
        s += "\n"
        s += tabulate(table, headers=headers, tablefmt="fancy_grid", floatfmt=".4f")

        logger.info(s)


if __name__ == "__main__":
    main()
