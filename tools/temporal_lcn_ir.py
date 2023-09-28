import argparse
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

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
parser.add_argument("-p", "--patch", type=int, required=True)
parser.add_argument("-t", "--threshold", type=float, required=True)
args = parser.parse_args()

x = np.linspace(0, 6, num=7, dtype=int)
x_avg = np.average(x)
x_avg_diff = x - x_avg


def get_smoothed_ir_pattern(diff: np.array, ks=9, threshold=0.005):
    diff = np.abs(diff)
    diff_avg = cv2.blur(diff, (ks, ks))
    ir = np.zeros_like(diff)
    ir[(diff / diff_avg) - 1 > threshold] = 1
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


def local_contrast_norm(image: np.ndarray, kernel_size=9, eps=1e-5, threshold=0.005):
    """compute local contrast normalization
    input:
        image: torch.tensor (height, width)
    output:
        normed_image
    """
    assert kernel_size % 2 == 1, "Kernel size should be odd"
    height, width = image.shape
    image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    unfold = nn.Unfold(kernel_size, padding=(kernel_size - 1) // 2)
    unfold_image = unfold(image)  # (batch, kernel_size*kernel_size, height*width)
    avg = torch.mean(unfold_image, dim=1).contiguous().view(1, 1, height, width)
    std = torch.std(unfold_image, dim=1, unbiased=False).contiguous().view(1, 1, height, width)
    normed_image = (image - avg) / (std + eps)

    normed_image = normed_image[0, 0].numpy()
    normed_image = np.clip(normed_image, -2, 2) / 2 * 127.5 + 127.5
    normed_image = normed_image.astype(np.uint8)

    return normed_image


def main():
    with open(args.split_file, "r") as f:
        prefix = [line.strip() for line in f]
    num = len(prefix)

    start = time.time()
    for idx in range(num):
        for direction in ["irL", "irR"]:
            p = prefix[idx]
            f0 = os.path.join(args.data_folder, p, f"1024_{direction}_real_off.png")
            f1 = os.path.join(args.data_folder, p, f"1024_{direction}_real_060.png")
            f2 = os.path.join(args.data_folder, p, f"1024_{direction}_real_120.png")
            f3 = os.path.join(args.data_folder, p, f"1024_{direction}_real_180.png")
            f4 = os.path.join(args.data_folder, p, f"1024_{direction}_real_240.png")
            f5 = os.path.join(args.data_folder, p, f"1024_{direction}_real_300.png")
            f6 = os.path.join(args.data_folder, p, f"1024_{direction}_real_360.png")
            img_0 = np.array(Image.open(f0).convert(mode="L"))
            img_1 = np.array(Image.open(f1).convert(mode="L"))
            img_2 = np.array(Image.open(f2).convert(mode="L"))
            img_3 = np.array(Image.open(f3).convert(mode="L"))
            img_4 = np.array(Image.open(f4).convert(mode="L"))
            img_5 = np.array(Image.open(f5).convert(mode="L"))
            img_6 = np.array(Image.open(f6).convert(mode="L"))
            h = img_0.shape[0]
            assert h in (540, 720, 1080), f"Illegal img shape: {img_0.shape}"
            if h in (720, 1080):
                img_0 = cv2.resize(img_0, (960, 540), interpolation=cv2.INTER_CUBIC)
                img_1 = cv2.resize(img_1, (960, 540), interpolation=cv2.INTER_CUBIC)
                img_2 = cv2.resize(img_2, (960, 540), interpolation=cv2.INTER_CUBIC)
                img_3 = cv2.resize(img_3, (960, 540), interpolation=cv2.INTER_CUBIC)
                img_4 = cv2.resize(img_4, (960, 540), interpolation=cv2.INTER_CUBIC)
                img_5 = cv2.resize(img_5, (960, 540), interpolation=cv2.INTER_CUBIC)
                img_6 = cv2.resize(img_6, (960, 540), interpolation=cv2.INTER_CUBIC)
            img_temp = np.concatenate(
                (
                    img_0[:, :, None],
                    img_1[:, :, None],
                    img_2[:, :, None],
                    img_3[:, :, None],
                    img_4[:, :, None],
                    img_5[:, :, None],
                    img_6[:, :, None],
                ),
                axis=-1,
            )

            # Get regression fit on temporal images
            print(f"Generating {p} temporal {direction} pattern {idx}/{num} time: {time.time() - start:.2f}s")
            h, w, d = img_temp.shape
            x = np.linspace(0, d - 1, num=d, dtype=int).reshape(1, 1, -1)
            x = np.repeat(x, h, axis=0)
            x = np.repeat(x, w, axis=1)  # [H, W, D]
            x_avg = np.average(x, axis=-1).reshape(h, w, 1)

            y = img_temp  # [H, W, D]
            y_avg = np.average(y, axis=-1).reshape(h, w, 1)

            numerator = np.sum((y - y_avg) * (x - x_avg), axis=-1)
            denominator = np.sum((x - x_avg) ** 2, axis=-1)  # [H, W]
            slope = numerator / denominator  # [H, W]
            slope = slope[:, :, None]
            intercept = y_avg - slope * x_avg
            img_temp_fit = slope * x + intercept

            # Get IR pattern
            diff = img_temp_fit[:, :, -1] - img_temp_fit[:, :, 0]
            # Normalize to [0,1]
            diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff))
            temp_pattern = get_smoothed_ir_pattern(diff, ks=args.patch, threshold=args.threshold)
            kernel = np.ones((args.patch, args.patch))
            temp_valid_mask = cv2.dilate(temp_pattern, kernel)
            cv2.imwrite(
                os.path.join(
                    args.data_folder, p, f"1024_{direction}_real_templcn2_ps{args.patch}_t{args.threshold}M.png"
                ),
                (temp_valid_mask * 255).astype(np.uint8),
            )
            lcn_pattern = local_contrast_norm(img_6, kernel_size=args.patch)
            temp_lcn_pattern = lcn_pattern * (1 - temp_valid_mask) + temp_pattern * 255
            temp_lcn_pattern = np.clip(temp_lcn_pattern, 0, 255).astype(np.uint8)

            # Save extracted IR pattern
            cv2.imwrite(
                os.path.join(
                    args.data_folder, p, f"1024_{direction}_real_templcn2_ps{args.patch}_t{args.threshold}.png"
                ),
                temp_lcn_pattern,
            )
            # cv2.imwrite(
            #     os.path.join(args.data_folder, p, f"1024_{direction}_real_templcn2_ps{args.patch}B.png"),
            #     (pattern > 127).astype(np.uint8) * 255,
            # )


if __name__ == "__main__":
    main()
