# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified by Jiayuan Gu
import sys
import time

from loguru import logger
from path import Path


def setup_logger(name, save_dir, rank, filename="log.train.txt", use_std=True):
    """Setup loguru logger"""
    # don't log results for the non-master process
    logger.remove()
    if rank > 0:
        return logger

    # TODO:How to set extra parammeter with defualt value in format?
    fmt = (
        f"<green>{{time:YYYY-MM-DD HH:mm:ss.SSS}}</green> | "
        f"<cyan>{name}</cyan> | "
        f"<lvl>{{level}}</lvl> | "
        f"<lvl>{{message}}</lvl>"
    )

    # logger to file
    if save_dir:
        Path(save_dir).makedirs_p()
        log_file = Path(save_dir) / filename
        if "log.test" in filename:
            with open(log_file, "w") as f:
                f.write("# coding=ASCII\n")
        logger.add(log_file, format=fmt)

    # logger to std stream
    if use_std:
        logger.add(sys.stdout, format=fmt)

    return logger
