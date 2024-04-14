# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_maskformer2_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    # NOTE: configs from original maskformer
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0
    cfg.MODEL.MASK_FORMER.GATE_WEIGHT = 1.0

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    # gating function
    cfg.MODEL.GATING = CN()
    cfg.MODEL.GATING.TRAIN_FOLDER = "/net/acadia2a/data/myao/evals/PQs/train"
    cfg.MODEL.GATING.TRAIN_FILES = ["105_L2", "105_L3", "105_L4", "105_L5", "105_L6"]
    cfg.MODEL.GATING.TRAIN_THRESHOLD = 0.65
    # cfg.MODEL.GATING.INFERENCE_THRESHOLD = 0.5
    cfg.MODEL.GATING.NUM_GATES = 4
    cfg.MODEL.GATING.TRAIN_COEFF = 0.01

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False


    # Added by AA
    # Training tricks # added for Lite-M2F (aka. skip scale)
    cfg.MODEL_EMA = CN()
    cfg.MODEL_EMA.DECAY = 0.999
    cfg.MODEL_EMA.DEVICE = ''
    cfg.MODEL_EMA.ENABLED = True
    cfg.MODEL_EMA.USE_EMA_WEIGHTS_FOR_EVAL_ONLY = True
    
    cfg.MODEL.MODEL_EMA = CN()
    cfg.MODEL.MODEL_EMA.ENABLED = False
    cfg.MODEL.MASK_FORMER.USE_FOCAL_LOSS = False
    cfg.MODEL.MASK_FORMER.FOCAL_ALPHA = 0.25
    cfg.MODEL.PYRAMID = CN()
    cfg.MODEL.PYRAMID.RESIDUAL = False
    cfg.MODEL.PYRAMID.PYRAMID_NUM_LAYERS = [1, 1, 1]
    cfg.MODEL.PYRAMID.IS_EXCITE_CONV = False
    cfg.MODEL.PYRAMID.MASK_OUTPUT_OPS = 'maxpool'
    cfg.MODEL.PYRAMID.EXCITE_TEMP = 1.0    
    cfg.MODEL.RT_DETR_ENCODER = CN()
    cfg.MODEL.RT_DETR_ENCODER.IN_CHANNELS = [192, 384, 768]
    cfg.MODEL.RT_DETR_ENCODER.RT_DETR_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.RT_DETR_ENCODER.RT_DETR_FEAT_STRIDES = [8, 16, 32]
    cfg.MODEL.RT_DETR_ENCODER.RT_DETR_HIDDEN_DIM = 256
    cfg.MODEL.RT_DETR_ENCODER.RT_DETR_USE_ENCODER_IDX = [2]
    cfg.MODEL.RT_DETR_ENCODER.RT_DETR_NHEAD = 8
    cfg.MODEL.RT_DETR_ENCODER.RT_DETR_DIM_FEEDFORWARD = 1024
    cfg.MODEL.RT_DETR_ENCODER.RT_DETR_DROPOUT = 0.
    cfg.MODEL.RT_DETR_ENCODER.RT_DETR_NUM_ENC_LAYERS = 1
    cfg.MODEL.RT_DETR_ENCODER.RT_DETR_ENC_ACT = 'gelu'
    cfg.MODEL.RT_DETR_ENCODER.RT_DETR_PE_TEMP = 10000
    cfg.MODEL.RT_DETR_ENCODER.RT_DETR_EXPANSION = 1.0
    cfg.MODEL.RT_DETR_ENCODER.RT_DETR_DEPTH_MULT = 1
    cfg.MODEL.RT_DETR_ENCODER.RT_DETR_ACT = 'silu'
    cfg.MODEL.RT_DETR_ENCODER.RT_DETR_EVAL_SIZE = [640, 640]
    cfg.MODEL.RT_DETR_ENCODER.RT_DETR_MASK_DIM = 256
    cfg.MODEL.SWIN.OUT_INDICES = (0, 1, 2, 3)
    # SimMIMSwin transformer backbone
    cfg.MODEL.SimMIMSwin = CN()
    cfg.MODEL.SimMIMSwin.PRETRAIN_IMG_SIZE = 256
    cfg.MODEL.SimMIMSwin.PATCH_SIZE = 4
    cfg.MODEL.SimMIMSwin.EMBED_DIM = 96
    cfg.MODEL.SimMIMSwin.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SimMIMSwin.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SimMIMSwin.WINDOW_SIZE = 7
    cfg.MODEL.SimMIMSwin.MLP_RATIO = 4.0
    cfg.MODEL.SimMIMSwin.DROP_RATE = 0.0
    cfg.MODEL.SimMIMSwin.DROP_PATH_RATE = 0.3
    cfg.MODEL.SimMIMSwin.APE = False
    cfg.MODEL.SimMIMSwin.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SimMIMSwin.OUT_INDICES = [1, 3, 9, 11]
    cfg.MODEL.SimMIMSwin.PATCH_NORM = True
    cfg.MODEL.SimMIMSwin.IMG_SIZE = 1024
    cfg.MODEL.SimMIMSwin.FPN_OUT_DIM = 768



    # HEIRA transformer backbone
    cfg.MODEL.HEIRA = CN()
    cfg.MODEL.HEIRA.INPUT_SIZE = (512, 1024) # (224, 224)
    cfg.MODEL.HEIRA.EMBED_DIM = 96
    cfg.MODEL.HEIRA.STAGES = [1, 2, 7, 2]
    cfg.MODEL.HEIRA.Q_POOL = 3
    cfg.MODEL.HEIRA.Q_STRIDE = (2, 2)
    cfg.MODEL.HEIRA.MASK_UNIT_SIZE = (8, 8)
    cfg.MODEL.HEIRA.MASK_UNIT_ATTN = (True, True, False, False)
    cfg.MODEL.HEIRA.DIM_MUL = 2.0
    cfg.MODEL.HEIRA.HEAD_MUL= 2.0
    cfg.MODEL.HEIRA.MLP_RATIO = 4.0
    cfg.MODEL.HEIRA.DROP_PATH_RATE = 0.0
    cfg.MODEL.HEIRA.patch_kernel = (7, 7)
    cfg.MODEL.HEIRA.patch_stride = (4, 4)
    cfg.MODEL.HEIRA.patch_padding = (3, 3)
    cfg.MODEL.HEIRA.OUT_FEATURES = ['stage2', 'stage3', 'stage4']

    # # SimMIMSwin transformer backbone
    # # cfg.MODEL.SimMIMSwin = CN()
    # cfg.MODEL.SimMIMSwin.IMG_SIZE = 1024
    # cfg.MODEL.SimMIMSwin.PATCH_SIZE = 4
    # cfg.MODEL.SimMIMSwin.EMBED_DIM = 96
    # cfg.MODEL.SimMIMSwin.DEPTHS = [2, 2, 6, 2]
    # cfg.MODEL.SimMIMSwin.NUM_HEADS = [3, 6, 12, 24]
    # # cfg.MODEL.SimMIMSwin.STRIDE = 16
    # cfg.MODEL.SimMIMSwin.WINDOW_SIZE = 16
    # cfg.MODEL.SimMIMSwin.MLP_RATIO = 4.0
    # cfg.MODEL.SimMIMSwin.DROP_PATH_RATE = 0.1
    # cfg.MODEL.SimMIMSwin.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    # cfg.MODEL.SimMIMSwin.FPN_OUT_DIM = 768



    # MViT2 backbone
    cfg.MODEL.MViT = CN()
    cfg.MODEL.MViT.EMBED_DIM = 96
    cfg.MODEL.MViT.DEPTH = 10
    cfg.MODEL.MViT.NUM_HEADS = 1
    cfg.MODEL.MViT.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.MViT.PATCH_KERNEL = (7, 7)
    cfg.MODEL.MViT.PATCH_STRIDE = (4, 4)
    cfg.MODEL.MViT.MLP_RATIO = 4.0
    cfg.MODEL.MViT.ADAPTIVE_KV_STRIDE = 4
    cfg.MODEL.MViT.QKV_POOL_KERNEL = (3, 3)
    cfg.MODEL.MViT.PATCH_PADDING = (3, 3)
    cfg.MODEL.MViT.DROP_PATH_RATE = 0.3
    cfg.MODEL.MViT.ADAPTIVE_WINDOW_SIZE = 56
    cfg.MODEL.MViT.LAST_BLOCK_INDEXES = (0, 2, 7, 9)
    cfg.MODEL.MViT.RESIDUAL_POOLING = True
    cfg.MODEL.MViT.USE_ABS_POS = False
    cfg.MODEL.MViT.USE_REL_POS = True
    cfg.MODEL.MViT.REL_POS_ZERO_INIT = True

    # swin-V2 transformer backbone
    cfg.MODEL.SWIN2 = CN()
    cfg.MODEL.SWIN2.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN2.PATCH_SIZE = 4
    cfg.MODEL.SWIN2.IN_CHANS = 3
    cfg.MODEL.SWIN2.NUM_CLASSES = 1000 # not in official swinV2
    cfg.MODEL.SWIN2.EMBED_DIM = 96
    cfg.MODEL.SWIN2.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN2.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN2.WINDOW_SIZE = 7
    cfg.MODEL.SWIN2.MLP_RATIO = 4.0
    cfg.MODEL.SWIN2.QKV_BIAS = True
    cfg.MODEL.SWIN2.QK_SCALE = None
    cfg.MODEL.SWIN2.DROP_RATE = 0.0
    cfg.MODEL.SWIN2.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN2.DROP_PATH_RATE = 0.1 # 0.3 for V1
    cfg.MODEL.SWIN2.APE = False
    cfg.MODEL.SWIN2.PATCH_NORM = True
    cfg.MODEL.SWIN2.PRETRAINED_WINDOW_SIZES = [0, 0, 0, 0]
    cfg.MODEL.SWIN2.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN2.USE_CHECKPOINT = False


    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75
