import argparse
import os
import os.path as osp
import sys

_ROOT_DIR = os.path.abspath(osp.join(osp.dirname(__file__), ".."))
sys.path.insert(0, _ROOT_DIR)
import time

import cv2
import numpy as np
from loguru import logger
from path import Path

from active_zero2.utils.geometry import cal_normal_map
from active_zero2.utils.io import load_pickle

parser = argparse.ArgumentParser(description="Extract temporal IR pattern from temporal real images")
parser.add_argument(
    "-s",
    "--split-file",
    type=str,
    metavar="FILE",
    required=True,
)
parser.add_argument(
    "-d",
    "--data-folder",
    type=str,
    required=True,
)

args = parser.parse_args()


def sub_main(prefix_list):
    n = len(prefix_list)
    start = time.time()
    for idx in range(n):
        p = prefix_list[idx]
        if os.path.exists(os.path.join(args.data_folder, p, "normalL.png")):
            try:
                normal_map = cv2.imread(os.path.join(args.data_folder, p, "normalL.png"), cv2.IMREAD_UNCHANGED)
                normal_map = (normal_map.astype(float)) / 1000.0
            except:
                logger.info(f"{os.path.join(args.data_folder, p, 'normalL.png')} is corrupted.")
                depth = cv2.imread(os.path.join(args.data_folder, p, "depthL.png"), cv2.IMREAD_UNCHANGED)
                depth = (depth.astype(float)) / 1000
                meta = load_pickle(os.path.join(args.data_folder, p, "meta.pkl"))
                intrinsic_l = meta["intrinsic_l"]
                normal_map = cal_normal_map(depth, intrinsic_l)
                normal_map_colored = ((normal_map + 1) * 127.5).astype(np.uint8)
                cv2.imwrite(os.path.join(args.data_folder, p, "normalL_colored.png"), normal_map_colored)
                normal_map = ((normal_map + 1) * 1000).astype(np.uint16)
                cv2.imwrite(os.path.join(args.data_folder, p, "normalL.png"), normal_map)
        else:
            logger.warning(f"No normal map in {os.path.join(args.data_folder, p)}")
        logger.info(f"Generating {p} normal map {idx}/{n} time: {time.time() - start:.2f}s")


def main():
    target_root = args.data_folder[: -len(args.data_folder.split("/")[-1])]
    timestamp = time.strftime("%y-%m-%d_%H-%M-%S")
    name = "normal_" + target_root.split("/")[-1]
    filename = f"log.check_normal.{timestamp}.txt"
    # set up logger
    logger.remove()
    fmt = (
        f"<green>{{time:YYYY-MM-DD HH:mm:ss.SSS}}</green> | "
        f"<cyan>{name}</cyan> | "
        f"<lvl>{{level}}</lvl> | "
        f"<lvl>{{message}}</lvl>"
    )
    # logger to file
    log_file = Path(target_root) / filename
    logger.add(log_file, format=fmt)

    # logger to std stream
    logger.add(sys.stdout, format=fmt)
    logger.info(f"Args: {args}")
    with open(args.split_file, "r") as f:
        prefix = [line.strip() for line in f]

    sub_main(prefix)


if __name__ == "__main__":
    main()
