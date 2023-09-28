# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified by Jiayuan Gu
from __future__ import division

from collections import defaultdict, deque

import numpy as np
import torch


class Metric(object):
    def update(self, *args, **kwargs):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    @property
    def result(self):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()

    @property
    def summary_str(self):
        raise NotImplementedError()


class Average(Metric):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    default_fmt = "{avg:.4f} ({global_avg:.4f})"
    default_summary_fmt = "{global_avg:.4f}"

    def __init__(self, window_size=20, fmt=None, summary_fmt=None):
        self.values = deque(maxlen=window_size)
        self.counts = deque(maxlen=window_size)
        self.sum = 0.0
        self.count = 0
        self.fmt = fmt or self.default_fmt
        self.summary_fmt = summary_fmt or self.default_summary_fmt

    def update(self, value, count=1):
        self.values.append(value)
        self.counts.append(count)
        self.sum += value
        self.count += count

    def reset(self):
        self.values.clear()
        self.counts.clear()
        self.sum = 0.0
        self.count = 0

    @property
    def result(self):
        return self.global_avg

    def __str__(self):
        return self.fmt.format(avg=self.avg, global_avg=self.global_avg)

    @property
    def summary_str(self):
        return self.summary_fmt.format(global_avg=self.global_avg)

    @property
    def avg(self):
        return np.sum(self.values) / np.sum(self.counts)

    @property
    def global_avg(self):
        return self.sum / self.count if self.count != 0 else float("nan")


class Accuracy(Average):
    default_fmt = "{avg:.2f} ({global_avg:.2f})"
    default_summary_fmt = "{global_avg:.2f}"

    def update(self, y_pred, y_true, ignore_mask=None):
        if ignore_mask == None:
            assert y_pred.shape == y_true.shape
            if torch.is_tensor(y_pred) and torch.is_tensor(y_true):
                mask = torch.eq(y_pred, y_true)
                value = mask.float().sum().item()
                count = mask.numel()
            elif isinstance(y_pred, np.ndarray) and isinstance(y_true, np.ndarray):
                mask = np.equal(y_pred, y_true)
                value = mask.sum().item()
                count = mask.size
            else:
                raise TypeError("{}, {}".format(type(y_pred), type(y_true)))
        else:
            assert y_pred.shape == y_true.shape
            assert y_true.shape == ignore_mask.shape
            if torch.is_tensor(y_pred) and torch.is_tensor(y_true):
                eq_mask = torch.eq(y_pred, y_true).float()
                eq_mask[ignore_mask] = 0.0
                value = eq_mask.float().sum().item()
                count = (1.0 - ignore_mask.float()).sum().item()
            else:
                raise TypeError("{}, {}".format(type(y_pred), type(y_true)))
        super().update(value=value, count=count)

    @property
    def avg(self):
        return super().avg * 100.0

    @property
    def global_avg(self):
        return super().global_avg * 100.0


class Precision(Average):
    default_fmt = "{avg:.2f} ({global_avg:.2f})"
    default_summary_fmt = "{global_avg:.2f}"

    def update(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape
        if torch.is_tensor(y_pred) and torch.is_tensor(y_true):
            mask = torch.logical_and(y_pred > 0, y_true > 0)
            value = mask.float().sum().item()
            count = (y_pred > 0).float().sum().item()
        elif isinstance(y_pred, np.ndarray) and isinstance(y_true, np.ndarray):
            mask = np.logical_and(y_pred > 0, y_true > 0)
            value = mask.sum().item()
            count = (y_pred > 0).sum().item()
        else:
            raise TypeError("{}, {}".format(type(y_pred), type(y_true)))
        super().update(value=value, count=count)

    @property
    def avg(self):
        return super().avg * 100.0

    @property
    def global_avg(self):
        return super().global_avg * 100.0


class Recall(Average):
    default_fmt = "{avg:.2f} ({global_avg:.2f})"
    default_summary_fmt = "{global_avg:.2f}"

    def update(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape
        if torch.is_tensor(y_pred) and torch.is_tensor(y_true):
            mask = torch.logical_and(y_pred > 0, y_true > 0)
            value = mask.float().sum().item()
            count = (y_true > 0).float().sum().item()
        elif isinstance(y_pred, np.ndarray) and isinstance(y_true, np.ndarray):
            mask = np.logical_and(y_pred > 0, y_true > 0)
            value = mask.sum().item()
            count = (y_true > 0).sum().item()
        else:
            raise TypeError("{}, {}".format(type(y_pred), type(y_true)))
        super().update(value=value, count=count)

    @property
    def avg(self):
        return super().avg * 100.0

    @property
    def global_avg(self):
        return super().global_avg * 100.0


class MetricLogger(object):
    """Metric logger."""

    def __init__(self, delimiter="\t"):
        self.metrics = defaultdict(Average)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                count = v.numel()
                value = v.item() if count == 1 else v.sum().item()
            elif isinstance(v, np.generic):
                count = v.size
                value = v.item() if count == 1 else v.sum().item()
            elif isinstance(v, (tuple, list)):
                value, count = v
                value = value.item()
                count = count.item()
            elif isinstance(v, (float, int)):
                value = v
                count = 1
            else:
                raise TypeError("Unsupported type: ".format(type(v)))
            self.metrics[k].update(value, count)

    def __getitem__(self, item):
        return self.metrics[item]

    def __str__(self):
        ret_str = []
        for name, metric in self.metrics.items():
            ret_str.append("{}: {}".format(name, str(metric)))
        return self.delimiter.join(ret_str)

    @property
    def summary_str(self):
        ret_str = []
        for name, metric in self.metrics.items():
            ret_str.append("{}: {}".format(name, metric.summary_str))
        return self.delimiter.join(ret_str)

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()


def test_Accuracy():
    acc_metric = Accuracy()
    acc_metric.update(np.array([1, 0, 1]), np.array([1, 0, 0]))
    np.testing.assert_allclose(acc_metric.result, 2.0 / 3.0 * 100.0)
    print(acc_metric)
    acc_metric.update(torch.tensor([1, 0, 1]), torch.tensor([1, 0, 1]))
    np.testing.assert_allclose(acc_metric.result, 5.0 / 6.0 * 100.0)
    print(acc_metric)
