# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detectron2/blob/main/tools/analyze_model.py

# fmt: off
import os
os.environ["DETECTRON2_DATASETS"] = '/net/acadia1b/data/samuel' # CityScapes
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import logging
import numpy as np
from collections import Counter
import tqdm
from fvcore.nn import flop_count_table  # can also try flop_count_str

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, LazyConfig, get_cfg, instantiate
from detectron2.data import build_detection_test_loader
from detectron2.engine import default_argument_parser
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.analysis import (
    FlopCountAnalysis,
    activation_count_operators,
    parameter_count_table,
)
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
import torch

import time

logger = logging.getLogger("detectron2")


def setup(args):
    os.environ["DETECTRON2_DATASETS"] = '/net/acadia1b/data/samuel' # CityScapes
    if args.config_file.endswith(".yaml"):
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(args.config_file)
        cfg.DATALOADER.NUM_WORKERS = 0
        cfg.merge_from_list(args.opts)
        cfg.freeze()
    else:
        cfg = LazyConfig.load(args.config_file)
        cfg = LazyConfig.apply_overrides(cfg, args.opts)
    setup_logger(name="fvcore")
    setup_logger()
    return cfg


def do_flop(cfg):
    if isinstance(cfg, CfgNode):
        data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
        model = build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    else:
        data_loader = instantiate(cfg.dataloader.test)
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    model.eval()

    counts = Counter()
    total_flops = []
    for idx, data in zip(tqdm.trange(args.num_inputs), data_loader):  # noqa
        if args.use_fixed_input_size and isinstance(cfg, CfgNode):
            # crop_size = cfg.INPUT.CROP.SIZE[0]
            # data[0]["image"] = torch.zeros((3, crop_size, crop_size))
            data[0]["image"] = torch.zeros((3, W, H))
            data[0]['width'], data[0]['height'] = W, H

        flops = FlopCountAnalysis(model, data)
        if idx > 0:
            flops.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        counts += flops.by_operator()
        total_flops.append(flops.total())

    logger.info("Flops table computed from only one input sample:\n" + flop_count_table(flops))
    logger.info(
        "Average GFlops for each type of operators:\n"
        + str([(k, v / (idx + 1) / 1e9) for k, v in counts.items()])
    )
    logger.info(
        "Total GFlops: {:.3f}±{:.3f}".format(np.mean(total_flops) / 1e9, np.std(total_flops) / 1e9)
    )


def do_fps(cfg):
    if isinstance(cfg, CfgNode):
        data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
        model = build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    else:
        data_loader = instantiate(cfg.dataloader.test)
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    model.eval()

    # logger.info("Using data_loader")
    # for i in range(args.warmup):
    #     total_times = []
    #     for idx, data in zip(range(args.num_inputs), data_loader):  # noqa
    #         if args.use_fixed_input_size and isinstance(cfg, CfgNode):
    #             data[0]["image"] = torch.zeros((3, W, H))
    #             data[0]['width'], data[0]['height'] = W, H

    #         # if i == args.warmup-1:
    #         start_time = time.time()
    #         # print('image size: ', str(list(data[0]['image'].size())))
    #         model(data)
    #         runtime = time.time() - start_time
    #         total_times.append(runtime)
    #     logger.info(
    #         "Average time per img: {:.3f}±{:.3f}s => throughput: {:.3f} imgs/s".format(np.mean(total_times), np.std(total_times), 1.0 / np.mean(total_times))
    #     )
    # logger.info("Without data_loader")


    for i in range(args.warmup):
        total_times = []

        for _ in range(args.num_inputs):
            data = [{
                "image": torch.zeros((3, W, H)),
                "width": W,
                "height": H
            }]
            start_time = time.time()
            model(data)
            runtime = time.time() - start_time
            total_times.append(runtime)
        logger.info(
            "Average time per img: {:.3f}±{:.3f}s => throughput: {:.3f} imgs/s".format(np.mean(total_times), np.std(total_times), 1.0 / np.mean(total_times))
        )


    logger.info('image size: ' + str(list(data[0]['image'].size())))


def do_activation(cfg):
    if isinstance(cfg, CfgNode):
        data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
        model = build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    else:
        data_loader = instantiate(cfg.dataloader.test)
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    model.eval()

    counts = Counter()
    total_activations = []
    for idx, data in zip(tqdm.trange(args.num_inputs), data_loader):  # noqa
        count = activation_count_operators(model, data)
        counts += count
        total_activations.append(sum(count.values()))
    logger.info(
        "(Million) Activations for Each Type of Operators:\n"
        + str([(k, v / idx) for k, v in counts.items()])
    )
    logger.info(
        "Total (Million) Activations: {}±{}".format(
            np.mean(total_activations), np.std(total_activations)
        )
    )


def do_parameter(cfg):
    if isinstance(cfg, CfgNode):
        model = build_model(cfg)
    else:
        model = instantiate(cfg.model)
    logger.info("Parameter Count:\n" + parameter_count_table(model, max_depth=5))


def do_structure(cfg):
    if isinstance(cfg, CfgNode):
        model = build_model(cfg)
    else:
        model = instantiate(cfg.model)
    logger.info("Model Structure:\n" + str(model))

if __name__ == "__main__":
    # W, H = 512, 512
    W, H = 512, 1024
    # W, H = 1024, 2048

    parser = default_argument_parser(
        epilog="""
Examples:
To show parameters of a model:
$ ./analyze_model.py --tasks parameter \\
    --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml
Flops and activations are data-dependent, therefore inputs and model weights
are needed to count them:
$ ./analyze_model.py --num-inputs 100 --tasks flop \\
    --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \\
    MODEL.WEIGHTS /path/to/model.pkl
"""
    )
    parser.add_argument(
        "--tasks",
        choices=["flop", "activation", "parameter", "structure", "fps"],
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "-n",
        "--num-inputs",
        default=100,
        type=int,
        help="number of inputs used to compute statistics for flops/activations, "
        "both are data dependent.",
    )
    parser.add_argument(
        "--use-fixed-input-size",
        action="store_true",
        help="use fixed input size when calculating flops",
    )
    parser.add_argument(
        "--warmup",
        default=20,
        type=int,
        help="the number of warm-ups before measuring the time of each module",
    )
    args = parser.parse_args()
    assert not args.eval_only
    assert args.num_gpus == 1

    cfg = setup(args)

    for task in args.tasks:
        {
            "flop": do_flop,
            "activation": do_activation,
            "parameter": do_parameter,
            "structure": do_structure,
            "fps": do_fps,
        }[task](cfg)
