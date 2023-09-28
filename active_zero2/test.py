#!/usr/bin/env python
import os
import os.path as osp
import sys

import tensorboardX

_ROOT_DIR = os.path.abspath(osp.join(osp.dirname(__file__), ".."))
sys.path.insert(0, _ROOT_DIR)

import argparse
import gc
import time
import warnings
import numpy as np

import torch
from torch.utils.data import DataLoader

from active_zero2.config import cfg
from active_zero2.datasets.build_dataset import build_dataset
from active_zero2.models.build_model import build_model
from active_zero2.utils.cfg_utils import purge_cfg
from active_zero2.utils.checkpoint import CheckpointerV2
from active_zero2.utils.loguru_logger import setup_logger
from active_zero2.utils.metric_logger import MetricLogger
from active_zero2.utils.metrics import ErrorMetric
from active_zero2.utils.reduce import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description="ActiveZero2 Evaluation")
    parser.add_argument(
        "--cfg",
        dest="config_file",
        default="",
        help="path to config file",
        type=str,
    )
    parser.add_argument("-s", "--save-file", action="store_true")
    parser.add_argument("-o", "--only", action="store_true", help="only save prediction results")
    parser.add_argument("-t", "--train", action="store_true", help="only save prediction for training")
    parser.add_argument("-r", "--realsense-mask", action="store_true", help="whether use realsense mask")
    parser.add_argument("-w", "--overwrite", action="store_true", help="overwrite existing results")
    parser.add_argument("-n", "--normal", action="store_true", help="generate real normal data")
    parser.add_argument("--normal-suffix", type=str, default="", help="real normal data name suffix")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # ---------------------------------------------------------------------------- #
    # Setup the experiment
    # ---------------------------------------------------------------------------- #
    args = parse_args()
    # Load the configuration
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    cfg.freeze()
    config_name = args.config_file.split("/")[-1].split(".")[0]

    # run name
    timestamp = time.strftime("%y-%m-%d_%H-%M-%S")
    run_name = "{:s}".format(timestamp)

    # Parse the output directory
    output_dir = cfg.OUTPUT_DIR
    # Replace '@' with config path
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        output_dir = output_dir.replace("@", config_path.replace("configs", "outputs"))
        if osp.isdir(output_dir):
            warnings.warn("Output directory exists.")
        os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger(f"ActiveZero2.test [{config_name}]", output_dir, rank=0, filename=f"log.test.{run_name}.txt")
    logger.info(args)
    from active_zero2.utils.collect_env import collect_env_info

    logger.info("Collecting env info (might take some time)\n" + collect_env_info())
    logger.info(f"Loaded config file: '{args.config_file}'")
    logger.info(f"Running with configs:\n{cfg}")
    # ---------------------------------------------------------------------------- #
    # Build models, optimizer, scheduler, checkpointer, etc.
    # ---------------------------------------------------------------------------- #
    # Build model
    set_random_seed(cfg.RNG_SEED)
    model = build_model(cfg)
    logger.info("Build model:\n{}".format(str(model)))
    model = model.cuda()
    # Enable CUDNN benchmark
    torch.backends.cudnn.benchmark = True

    # Build checkpoint
    checkpointer = CheckpointerV2(model, save_dir=output_dir, logger=logger)
    if cfg.TEST.WEIGHT:
        weight_path = cfg.TEST.WEIGHT.replace("@", output_dir)
        checkpointer.load(weight_path, resume=False)
    else:
        checkpointer.load(None, resume=True)

    if args.save_file or args.train or args.only or args.normal:
        if cfg.TEST.WEIGHT:
            model_path = weight_path
        else:
            model_path = checkpointer.get_checkpoint_file()
        if model_path:
            model_name = model_path.split("/")[-1].split(".")[0]
            file_dir = osp.join(output_dir, model_name)
        else:
            file_dir = output_dir
        if osp.isdir(file_dir):
            logger.warning(f"File directory {file_dir} exists")
        os.makedirs(file_dir, exist_ok=True)
        logger.info(f"Save result visualization to {file_dir}")

    # Build data loader
    logger.info(f"Build dataloader")
    set_random_seed(cfg.RNG_SEED)
    test_sim_dataset = build_dataset(cfg, mode="test", domain="sim")
    test_real_dataset = build_dataset(cfg, mode="test", domain="real")

    assert cfg.TEST.BATCH_SIZE == 1, "Only support cfg.TEST.BATCH_SIZE=1"

    if test_sim_dataset:
        test_sim_loader = DataLoader(
            test_sim_dataset,
            batch_size=cfg.TEST.BATCH_SIZE,
            num_workers=cfg.TEST.NUM_WORKERS,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
        )

    else:
        test_sim_loader = None
    if test_real_dataset:
        test_real_loader = DataLoader(
            test_real_dataset,
            batch_size=cfg.TEST.BATCH_SIZE,
            num_workers=cfg.TEST.NUM_WORKERS,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
        )
    else:
        test_real_loader = None

    # Build metrics
    metric = ErrorMetric(
        model_type=cfg.MODEL_TYPE,
        use_mask=cfg.TEST.USE_MASK,
        max_disp=cfg.TEST.MAX_DISP,
        depth_range=cfg.TEST.DEPTH_RANGE,
        num_classes=cfg.DATA.NUM_CLASSES,
        is_depth=cfg.TEST.IS_DEPTH,
        realsense_mask=args.realsense_mask,
    )

    # ---------------------------------------------------------------------------- #
    # Test
    # ---------------------------------------------------------------------------- #
    model.eval()
    logger.info("Begin evaluation...")
    eval_tic = tic = time.time()
    if test_sim_loader:
        logger.info("Sim Evaluation")
        metric.reset()
        with torch.no_grad():
            for iteration, data_batch in enumerate(test_sim_loader):
                curr_dict = {}
                cur_iter = iteration + 1
                data_time = time.time() - tic
                data_dir = data_batch["dir"][0]
                if args.train and os.path.exists(
                        osp.join(file_dir, data_dir + "_sim_train", "depth_pred_rgb_colored.png")
                ):
                    if args.overwrite:
                        logger.info(f"iter: {cur_iter:5d} Overwrite {data_dir}")
                    else:
                        logger.info(f"iter: {cur_iter:5d} Skip {data_dir}")
                        continue
                data_batch = {
                    k: v.cuda(non_blocking=True) for k, v in data_batch.items() if isinstance(v, torch.Tensor)
                }
                data_batch["dir"] = data_dir
                # Forward
                pred_dict = model(data_batch)
                # left_right warp
                swap_data_batch = {
                    "img_l": torch.flip(data_batch["img_r"], (-1,)),
                    "img_r": torch.flip(data_batch["img_l"], (-1,)),
                }
                if "img_off_l" in data_batch:
                    swap_data_batch["img_off_l"] = torch.flip(data_batch["img_off_r"], (-1,))
                    swap_data_batch["img_off_r"] = torch.flip(data_batch["img_off_l"], (-1,))
                    swap_data_batch["img_edge_l"] = torch.flip(data_batch["img_edge_r"], (-1,))
                    swap_data_batch["img_edge_r"] = torch.flip(data_batch["img_edge_l"], (-1,))
                swap_pred_dict = model(swap_data_batch)
                if cfg.MODEL_TYPE in (
                        "PSMNet",
                        "PSMNetRange",
                        "PSMNetDilation",
                        "PSMNetKPAC",
                        "PSMNetGrad",
                        "PSMNetADV",
                        "PSMNetADV4",
                        "PSMNetGrad2DADV",
                ):
                    pred_dict["pred3_r"] = torch.flip(swap_pred_dict["pred3"], (-1,))
                elif cfg.MODEL_TYPE == "PSMNetDilationDF":
                    pred_dict["df_pred_r"] = torch.flip(swap_pred_dict["df_pred"], (-1,))
                elif cfg.MODEL_TYPE in ("PSMNetInpainting",):
                    pred_dict["pred3_r"] = torch.flip(swap_pred_dict["pred3"], (-1,))
                    pred_dict["inpaint_r"] = torch.flip(swap_pred_dict["inpaint"], (-1,))
                elif cfg.MODEL_TYPE == "PSMNetEdge":
                    pred_dict["pred3_r"] = torch.flip(swap_pred_dict["pred3"], (-1,))
                    pred_dict["edge_r"] = torch.flip(swap_pred_dict["edge"], (-1,))
                if cfg.MODEL_TYPE == "PSMNetGrad2DADV":
                    model.test_D(data_batch, pred_dict)
                elif cfg.MODEL_TYPE == "PSMNetEdgeNormal":
                    pred_dict["pred3_r"] = torch.flip(swap_pred_dict["pred3"], (-1,))
                    pred_dict["edge_r"] = torch.flip(swap_pred_dict["edge"], (-1,))
                if args.train:
                    metric.save_for_train(
                        data_batch, pred_dict, save_folder=osp.join(file_dir, data_dir + "_sim_train"), real_data=False
                    )
                elif args.only:
                    metric.save_preds_only(
                        data_batch, pred_dict, save_folder=osp.join(file_dir, data_dir + "_sim"), real_data=False
                    )
                else:
                    curr_dict.update(
                        metric.compute(
                            data_batch,
                            pred_dict,
                            save_folder=osp.join(file_dir, data_dir + "_sim") if args.save_file else "",
                            real_data=False,
                        )
                    )

                if cfg.MODEL_TYPE == "PSMNetEdge" and "img_disp_edge_l" in data_batch:
                    _, prec, recall = model.compute_edge_loss(data_batch, pred_dict)
                    curr_dict.update(edge_prec=prec.item(), edge_recall=recall.item())

                batch_time = time.time() - tic
                curr_dict.update(time=batch_time, data=data_time)
                # Logging
                logger.info(
                    "Sim Test "
                    + "\t".join(["iter: {iter:3d}", "scene: {scene}", "{meters}", "max mem: {memory:.0f}"]).format(
                        iter=cur_iter,
                        scene=data_dir,
                        meters="\t".join([f"{k}: {v:.3f}" for k, v in curr_dict.items()]),
                        memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                    )
                )

                tic = time.time()

        epoch_time_eval = time.time() - eval_tic
        logger.info("Sim Test total_time: {:.2f}s".format(epoch_time_eval))
        if not (args.only or args.train):
            logger.info("Sim Eval Metric: \n" + metric.summary())
    if test_real_loader:
        logger.info("Real Evaluation")
        forward_time_list = []
        eval_tic = tic = time.time()
        metric.reset()
        with torch.no_grad():
            for iteration, data_batch in enumerate(test_real_loader):
                curr_dict = {}
                cur_iter = iteration + 1
                data_time = time.time() - tic
                full_dir = data_batch["full_dir"][0]
                data_dir = data_batch["dir"][0]
                if args.train and os.path.exists(
                        osp.join(file_dir, data_dir + "_real_train", "depth_pred_rgb_colored.png")
                ):
                    if args.overwrite:
                        logger.info(f"iter: {cur_iter:5d} Overwrite {data_dir}")
                    else:
                        logger.info(f"iter: {cur_iter:5d} Skip {data_dir}")
                        continue
                data_batch = {
                    k: v.cuda(non_blocking=True) for k, v in data_batch.items() if isinstance(v, torch.Tensor)
                }
                data_batch["dir"] = data_dir
                # Forward
                forward_tic = time.time()
                pred_dict = model(data_batch)
                forward_time = time.time() - forward_tic
                forward_time_list.append(forward_time)
                # left_right warp
                swap_data_batch = {
                    "img_l": torch.flip(data_batch["img_r"], (-1,)),
                    "img_r": torch.flip(data_batch["img_l"], (-1,)),
                }
                if "img_off_l" in data_batch:
                    swap_data_batch["img_off_l"] = torch.flip(data_batch["img_off_r"], (-1,))
                    swap_data_batch["img_off_r"] = torch.flip(data_batch["img_off_l"], (-1,))
                    swap_data_batch["img_edge_l"] = torch.flip(data_batch["img_edge_r"], (-1,))
                    swap_data_batch["img_edge_r"] = torch.flip(data_batch["img_edge_l"], (-1,))
                swap_pred_dict = model(swap_data_batch)
                if cfg.MODEL_TYPE in (
                        "PSMNet",
                        "PSMNetRange",
                        "PSMNetDilation",
                        "PSMNetKPAC",
                        "PSMNetGrad",
                        "PSMNetADV",
                        "PSMNetADV4",
                        "PSMNetGrad2DADV",
                ):
                    pred_dict["pred3_r"] = torch.flip(swap_pred_dict["pred3"], (-1,))
                elif cfg.MODEL_TYPE == "PSMNetDilationDF":
                    pred_dict["df_pred_r"] = torch.flip(swap_pred_dict["df_pred"], (-1,))
                elif cfg.MODEL_TYPE == "PSMNetInpainting":
                    pred_dict["pred3_r"] = torch.flip(swap_pred_dict["pred3"], (-1,))
                    pred_dict["inpaint_r"] = torch.flip(swap_pred_dict["inpaint"], (-1,))
                elif cfg.MODEL_TYPE == "PSMNetEdge":
                    pred_dict["pred3_r"] = torch.flip(swap_pred_dict["pred3"], (-1,))
                    pred_dict["edge_r"] = torch.flip(swap_pred_dict["edge"], (-1,))
                    pred_dict["conf_map_r"] = torch.flip(swap_pred_dict["conf_map"], (-1,))
                elif cfg.MODEL_TYPE == "PSMNetEdgeNormal":
                    pred_dict["pred3_r"] = torch.flip(swap_pred_dict["pred3"], (-1,))
                    pred_dict["edge_r"] = torch.flip(swap_pred_dict["edge"], (-1,))

                if cfg.MODEL_TYPE == "PSMNetGrad2DADV":
                    model.test_D(data_batch, pred_dict)
                if args.train:
                    metric.save_for_train(
                        data_batch, pred_dict, save_folder=osp.join(file_dir, data_dir + "_real_train"), real_data=True
                    )
                elif args.only:
                    metric.save_preds_only(
                        data_batch, pred_dict, save_folder=osp.join(file_dir, data_dir + "_real"), real_data=True
                    )
                elif args.normal:
                    if cfg.MODEL_TYPE == "PSMNetClearGrasp":
                        metric.save_normal(
                            data_batch,
                            pred_dict,
                            save_folder=osp.join(file_dir, data_dir + "_real"),
                            suffix=args.normal_suffix,
                        )
                    else:
                        metric.save_normal(data_batch, pred_dict, save_folder=full_dir, suffix=args.normal_suffix)
                else:
                    curr_dict.update(
                        metric.compute(
                            data_batch,
                            pred_dict,
                            save_folder=osp.join(file_dir, data_dir + "_real") if args.save_file else "",
                            real_data=True,
                        )
                    )

                if cfg.MODEL_TYPE == "PSMNetEdge" and "img_disp_edge_l" in data_batch:
                    _, prec, recall = model.compute_edge_loss(data_batch, pred_dict)
                    curr_dict.update(edge_prec=prec.item(), edge_recall=recall.item())

                batch_time = time.time() - tic
                curr_dict.update(time=batch_time, data=data_time, inference=forward_time)
                # Logging
                logger.info(
                    "Real Test "
                    + "\t".join(["iter: {iter:3d}", "scene: {scene}", "{meters}", "max mem: {memory:.0f}"]).format(
                        iter=cur_iter,
                        scene=data_dir,
                        meters="\t".join([f"{k}: {v:.3f}" for k, v in curr_dict.items()]),
                        memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                    )
                )

                tic = time.time()

        # END
        epoch_time_eval = time.time() - eval_tic
        logger.info("Real Test forward_time: {:.4f}s ({:.4f})".format(np.mean(forward_time_list),
                                                                      np.std(forward_time_list, ddof=1)))
        logger.info("Real Test total_time: {:.2f}s".format(epoch_time_eval))
        if not (args.only or args.train):
            logger.info("Real Eval Metric: \n" + metric.summary())
        exit(0)
