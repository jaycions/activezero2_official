import argparse
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import multiprocessing

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


def local_contrast_norm(image: np.ndarray, kernel_size=9, eps=1e-5):
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


def sub_main(prefix_list):
    n = len(prefix_list)
    start = time.time()
    for idx in range(n):
        for direction in ["irL", "irR"]:
            p = prefix_list[idx]
            f6 = os.path.join(args.data_folder, p, f"0128_{direction}_kuafu_half.png")
            img_6 = np.array(Image.open(f6).convert(mode="L"))
            h = img_6.shape[0]
            assert h in (540, 720, 1080), f"Illegal img shape: {img_6.shape}"
            if h in (720, 1080):
                img_6 = cv2.resize(img_6, (960, 540), interpolation=cv2.INTER_CUBIC)

            print(f"Generating {p} LCN {direction} pattern {idx}/{n} time: {time.time() - start:.2f}s")
            lcn_pattern = local_contrast_norm(img_6, kernel_size=args.patch)
            cv2.imwrite(os.path.join(args.data_folder, p, f"0128_{direction}_lcn_ps{args.patch}.png"), lcn_pattern)


def main():
    with open(args.split_file, "r") as f:
        prefix = [line.strip() for line in f]
    num = len(prefix)
    assert num % args.mp == 0, f"num: {num}, mp: {args.mp}"
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
