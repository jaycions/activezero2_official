import argparse
import os
import time

import cv2
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description="Extract binary IR pattern from real images")
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
parser.add_argument("-t", "--threshold", type=float, required=True)
args = parser.parse_args()

x = np.linspace(0, 6, num=7, dtype=int)
x_avg = np.average(x)
x_avg_diff = x - x_avg


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


def get_regression(y):
    # total_num = len(y)
    # x = np.linspace(0, total_num-1, num=total_num, dtype=int)
    # x_avg = np.average(x)
    # x_avg_diff = x - x_avg
    y_avg = np.average(y)
    y_avg_diff = y - y_avg
    slope = np.sum(x_avg_diff * y_avg_diff) / np.sum(x_avg_diff**2)
    intercept = y_avg - slope * x_avg
    pred = slope * x + intercept
    return slope, pred


def main():
    with open(args.split_file, "r") as f:
        prefix = [line.strip() for line in f]
    num = len(prefix)

    start = time.time()
    for idx in range(num):
        for direction in ["irL", "irR"]:
            p = prefix[idx]
            f0 = os.path.join(args.data_folder, p, f"1024_{direction}_real_off.png")
            f6 = os.path.join(args.data_folder, p, f"1024_{direction}_real_360.png")
            img_0 = np.array(Image.open(f0).convert(mode="L"))
            img_6 = np.array(Image.open(f6).convert(mode="L"))
            h = img_0.shape[0]
            assert h in (540, 720, 1080), f"Illegal img shape: {img_0.shape}"
            if h in (720, 1080):
                img_0 = cv2.resize(img_0, (960, 540), interpolation=cv2.INTER_CUBIC)
                img_6 = cv2.resize(img_6, (960, 540), interpolation=cv2.INTER_CUBIC)

            print(f"Generating {p} binary {direction} pattern {idx}/{num} time: {time.time() - start:.2f}s")
            binary_pattern = get_smoothed_ir_pattern2(img_6, img_0, ks=args.patch, threshold=args.threshold)
            cv2.imwrite(
                os.path.join(args.data_folder, p, f"1024_{direction}_real_bin_ps{args.patch}_t{args.threshold}.png"),
                binary_pattern,
            )


if __name__ == "__main__":
    main()
