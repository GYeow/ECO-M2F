import os
os.environ["DETECTRON2_DATASETS"] = '/net/acadia1b/data/samuel' # CityScapes
# os.environ["DETECTRON2_DATASETS"] = '/net/acadia3a/data/acadia1a/samuel' # COCO

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
from train_net import *

logger = logging.getLogger("detectron2")



def get_args():
    parser = default_argument_parser()
    args = parser.parse_args()
    args.eval_only = True
    return args

def do_flop(args, cfg):
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

    args.num_inputs = len(data_loader)

    counts = Counter()
    total_flops = []
    total_flops_counter = Counter()
    for idx, data in zip(tqdm.trange(args.num_inputs), data_loader):  # noqa
        # if args.use_fixed_input_size and isinstance(cfg, CfgNode):
        #     # crop_size = cfg.INPUT.CROP.SIZE[0]
        #     # data[0]["image"] = torch.zeros((3, crop_size, crop_size))
        #     data[0]["image"] = torch.zeros((3, W, H))
        #     data[0]['width'], data[0]['height'] = W, H

        flops = FlopCountAnalysis(model, data)
        if idx > 0:
            flops.unsupported_ops_warnings(False).uncalled_modules_warnings(False)

        counts += flops.by_operator()
        total_flops.append(flops.total())
        total_flops_counter += flops.by_module()
    # logger.info("Flops table computed from only one input sample:\n" + flop_count_table(flops))
    import operator as op
    from tabulate import tabulate
    # print(tabulate(
    #     [[k, v/args.num_inputs/1e9] for k, v in total_flops_counter.items() if op.countOf(k, ".")<4], 
    #     headers=["module", "Gflops"], floatfmt=".4f"))
    print(tabulate(
        [[k, v/args.num_inputs/1e9] for k, v in total_flops_counter.items()], 
        headers=["module", "Gflops"], floatfmt=".4f"))

    logger.info(
        "Average GFlops for each type of operators:\n"
        + str([(k, v / (args.num_inputs) / 1e9) for k, v in counts.items()])
    )
    # logger.info(
    #     "Total GFlops: {:.3f}±{:.3f}".format(np.mean(total_flops) / 1e9, np.std(total_flops) / 1e9)
    # )
    logger.info(
        "Average GFlops: {:.3f}±{:.3f}".format(np.mean(total_flops) /1e9, np.std(total_flops)/ 1e9)
    )

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg

def eval_models():
    args = get_args()
    
    print("Command Line Args:", args)

    cfg = setup(args) # cfg is immutable

    do_flop(args, cfg)



if __name__ == "__main__":
    eval_models()