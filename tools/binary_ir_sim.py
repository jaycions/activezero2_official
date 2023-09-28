import argparse
import multiprocessing
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

parser = argparse.ArgumentParser(description="Extract LCN IR pattern from IR images")
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
parser.add_argument("-p", "--patch", type=int, required=True)
parser.add_argument("--mp", type=int, default=1, help="multi-process")
args = parser.parse_args()


def get_smoothed_ir_pattern2(img_ir: np.array, img: np.array, ks=11, threshold=0.005):
    h, w = img_ir.shape
    hs = int(h // ks)
    ws = int(w // ks)
    diff = np.abs(img_ir - img)
    diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff))
    diff_avg = cv2.resize(diff, (ws, hs), interpolation=cv2.INTER_AREA)
    diff_avg = cv2.resize(diff_avg, (w, h), interpolation=cv2.INTER_AREA)
    ir = np.zeros_like(diff)
    diff2 = diff - diff_avg
    ir[diff2 > threshold] = 1
    ir = (ir * 255).astype(np.uint8)
    return ir


def sub_main(prefix_list):
    n = len(prefix_list)
    start = time.time()
    for idx in range(n):
        for direction in ["irL", "irR"]:
            p = prefix_list[idx]
            f0 = os.path.join(args.data_folder, p, f"0128_{direction}_kuafu_half_no_ir.png")
            f6 = os.path.join(args.data_folder, p, f"0128_{direction}_kuafu_half.png")
            img_0 = np.array(Image.open(f0).convert(mode="L"))
            img_6 = np.array(Image.open(f6).convert(mode="L"))
            h = img_0.shape[0]
            assert h in (540, 720, 1080), f"Illegal img shape: {img_0.shape}"
            if h in (720, 1080):
                img_0 = cv2.resize(img_0, (960, 540), interpolation=cv2.INTER_CUBIC)
                img_6 = cv2.resize(img_6, (960, 540), interpolation=cv2.INTER_CUBIC)

            print(f"Generating {p} binary sim {direction} pattern {idx}/{n} time: {time.time() - start:.2f}s")
            binary_pattern = get_smoothed_ir_pattern2(img_6, img_0, ks=args.patch)
            cv2.imwrite(os.path.join(args.data_folder, p, f"0128_{direction}_bin_ps{args.patch}.png"), binary_pattern)


def main():
    print("Multi-processing: ", args.mp)
    with open(args.split_file, "r") as f:
        prefix = [line.strip() for line in f]
    num = len(prefix)
    assert num % args.mp == 0
    l = num // args.mp

    p_list = []
    for i in range(args.mp):
        p = multiprocessing.Process(target=sub_main, args=(prefix[i * l : (i + 1) * l],))
        p.start()
        p_list.append(p)

    for p in p_list:
        p.join()


if __name__ == "__main__":
    main()
