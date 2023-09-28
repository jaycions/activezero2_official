import random

import numpy as np
import torch
import torchvision.transforms as Transforms


def data_augmentation(data_aug_cfg=None, domain="sim"):
    """ """
    transform_list = [Transforms.ToTensor()]
    if data_aug_cfg and domain in data_aug_cfg.DOMAINS:
        if data_aug_cfg.GAUSSIAN_BLUR:
            gaussian_sig = random.uniform(data_aug_cfg.GAUSSIAN_MIN, data_aug_cfg.GAUSSIAN_MAX)
            transform_list += [Transforms.GaussianBlur(kernel_size=data_aug_cfg.GAUSSIAN_KERNEL, sigma=gaussian_sig)]
        if data_aug_cfg.COLOR_JITTER:
            transform_list += [
                Transforms.ColorJitter(
                    brightness=[data_aug_cfg.BRIGHT_MIN, data_aug_cfg.BRIGHT_MAX],
                    contrast=[data_aug_cfg.CONTRAST_MIN, data_aug_cfg.CONTRAST_MAX],
                    saturation=[data_aug_cfg.SATURATION_MIN, data_aug_cfg.SATURATION_MAX],
                    hue=[data_aug_cfg.HUE_MIN, data_aug_cfg.HUE_MAX],
                )
            ]
    # Normalization
    transform_list += [
        Transforms.Normalize(
            mean=[0.45],
            std=[0.224],
        )
    ]
    custom_augmentation = Transforms.Compose(transform_list)
    return custom_augmentation


class SimIRNoise(object):
    def __init__(self, data_aug_cfg=None, domain="sim"):
        if data_aug_cfg is not None and domain == "sim":
            self.sim_ir = data_aug_cfg.SIM_IR
            self.speckle_shape_min = data_aug_cfg.SPECKLE_SHAPE_MIN
            self.speckle_shape_max = data_aug_cfg.SPECKLE_SHAPE_MAX
            self.gaussian_mu = data_aug_cfg.GAUSSIAN_MU
            self.gaussian_sigma = data_aug_cfg.GAUSSIAN_SIGMA
        else:
            self.sim_ir = False

    def apply(self, img):
        if self.sim_ir:
            speckle_shape = np.random.uniform(self.speckle_shape_min, self.speckle_shape_max)
            img = img * np.random.gamma(shape=speckle_shape, scale=1.0 / speckle_shape, size=img.shape)
            img = img + self.gaussian_mu + self.gaussian_sigma * np.random.standard_normal(img.shape)
        return img
