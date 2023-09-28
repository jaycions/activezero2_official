# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified by Jiayuan Gu
import hashlib
import os

import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel


def get_md5(filename):
    hash_obj = hashlib.md5()
    with open(filename, "rb") as f:
        hash_obj.update(f.read())
    return hash_obj.hexdigest()


class Checkpointer(object):
    """Checkpoint the model and relevant states.

    Supported features:
    1. Resume optimizer and scheduler
    2. Automatically deal with DataParallel, DistributedDataParallel
    3. Resume last saved checkpoint

    """

    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        logger=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        # logging
        self.logger = logger
        self._print = logger.info if logger else print

    def save(self, name, tag=True, **kwargs):
        if not self.save_dir:
            return

        data = dict()
        if isinstance(self.model, (DataParallel, DistributedDataParallel)):
            data["model"] = self.model.module.state_dict()
        else:
            data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        torch.save(data, save_file)
        self._print("Saved checkpoint to {}, MD5: {}".format(os.path.abspath(save_file), get_md5(save_file)))
        if tag:
            self.tag_last_checkpoint(save_file)

    def load(self, path=None, resume=True, resume_states=True, strict=True):
        if resume and self.has_checkpoint():
            # override argument with existing checkpoint
            path = self.get_checkpoint_file()
        if not path:
            # no checkpoint could be found
            self._print("No checkpoint found. Initializing model from scratch")
            return {}

        self._print("Loading checkpoint from {}, MD5: {}".format(path, get_md5(path)))
        checkpoint = self._load_file(path)

        if isinstance(self.model, (DataParallel, DistributedDataParallel)):
            self.model.module.load_state_dict(checkpoint.pop("model"), strict=strict)
        else:
            self.model.load_state_dict(checkpoint.pop("model"), strict=strict)
        if resume_states:
            if "optimizer" in checkpoint and self.optimizer:
                self.logger.info("Loading optimizer from {}".format(path))
                self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
            if "scheduler" in checkpoint and self.scheduler:
                self.logger.info("Loading scheduler from {}".format(path))
                self.scheduler.load_state_dict(checkpoint.pop("scheduler"))
        else:
            checkpoint = {}

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
            # If not absolute path, add save_dir as prefix
            if not os.path.isabs(last_saved):
                last_saved = os.path.join(self.save_dir, last_saved)
        except IOError:
            # If file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        # If not absolute path, only save basename
        if not os.path.isabs(last_filename):
            last_filename = os.path.basename(last_filename)
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, path):
        return torch.load(path, map_location=torch.device("cpu"))


class CheckpointerV2(Checkpointer):
    """Support max_to_keep like tf.Saver"""

    def __init__(self, *args, max_to_keep=3, local_rank=0, **kwargs):
        super(CheckpointerV2, self).__init__(*args, **kwargs)
        self.max_to_keep = max_to_keep
        self.local_rank = local_rank
        self._last_checkpoints = []

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            self._last_checkpoints = self._load_last_checkpoints(save_file)
            last_saved = self._last_checkpoints[-1]
        except IOError:
            # If file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        # Remove first from list if the same name was used before.
        for path in self._last_checkpoints:
            if last_filename == path:
                self._last_checkpoints.remove(path)
        # Append new path to list
        self._last_checkpoints.append(last_filename)
        # If more than max_to_keep, remove the oldest.
        self._delete_old_checkpoint()
        # Dump last checkpoints to a file
        self._save_checkpoint_file(save_file)

    def _delete_old_checkpoint(self):
        if len(self._last_checkpoints) > self.max_to_keep:
            path = self._last_checkpoints.pop(0)
            try:
                os.remove(path)
            except Exception as e:
                self._print("Ignoring: %s", str(e))

    def _save_checkpoint_file(self, path):
        with open(path, "w") as f:
            lines = []
            for p in self._last_checkpoints:
                if not os.path.isabs(p):
                    # If not absolute path, only save basename
                    p = os.path.basename(p)
                lines.append(p)
            f.write("\n".join(lines))

    def _load_last_checkpoints(self, path):
        last_checkpoints = []
        with open(path, "r") as f:
            for p in f.readlines():
                p = p.rstrip()
                if not os.path.isabs(p):
                    # If not absolute path, add save_dir as prefix
                    p = os.path.join(self.save_dir, p)
                last_checkpoints.append(p)
        return last_checkpoints

    def save(self, name, tag=True, **kwargs):
        if not self.save_dir or self.local_rank != 0:
            return

        data = dict()
        if isinstance(self.model, (DataParallel, DistributedDataParallel)):
            data["model"] = self.model.module.state_dict()
        else:
            data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        torch.save(data, save_file)
        self._print("Saved checkpoint to {}, MD5: {}".format(os.path.abspath(save_file), get_md5(save_file)))
        if tag:
            self.tag_last_checkpoint(save_file)

    def load(self, path=None, resume=True, resume_states=True, strict=True):
        if resume and self.has_checkpoint():
            # override argument with existing checkpoint
            path = self.get_checkpoint_file()
        if not path:
            # no checkpoint could be found
            self._print("No checkpoint found. Initializing model from scratch")
            return {}

        self._print("Loading checkpoint from {}, MD5: {}".format(path, get_md5(path)))
        checkpoint = self._load_file(path)

        if isinstance(self.model, (DataParallel, DistributedDataParallel)):
            self.model.module.load_state_dict(checkpoint.pop("model"), strict=strict)
        else:
            self.model.load_state_dict(checkpoint.pop("model"), strict=strict)
        if resume_states:
            if "optimizer" in checkpoint and self.optimizer:
                self.logger.info("Loading optimizer from {}".format(path))
                self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
            if "scheduler" in checkpoint and self.scheduler:
                self.logger.info("Loading scheduler from {}".format(path))
                self.scheduler.load_state_dict(checkpoint.pop("scheduler"))
        else:
            checkpoint = {}

        # return any further checkpoint data
        return checkpoint

    def _load_file(self, path):
        # configure map_location properly
        map_location = {"cuda:%d" % 0: "cuda:%d" % self.local_rank}
        return torch.load(path, map_location=map_location)
