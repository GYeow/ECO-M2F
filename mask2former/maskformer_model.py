# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple
import os
import json
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher

import csv

@META_ARCH_REGISTRY.register()
class MaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        gating_files: Tuple[float],
        # gating_train_threshold: float,
        gating_train_coeff: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        gating_output_file: str
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        # self.gating_train_threshold = gating_train_threshold
        self.gating_train_coeff = gating_train_coeff
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        # self.gating_targets = self.prepare_gating_targets(gating_files, gating_train_threshold)
        self.gating_targets = self.prepare_gating_targets(gating_files, gating_train_coeff)

        self.gating_output_file = gating_output_file
        
        # # writing to csv file
        # with open(self.gating_output_file, 'w') as csvfile:
        #     # creating a csv dict writer object
        #     writer = csv.DictWriter(csvfile, fieldnames=['image_id', 'gates2', 'gates3', 'gates4', 'gates5', 'gates6'])
        #     # writing headers (field names)
        #     writer.writeheader()

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        loss_ce_gates = cfg.MODEL.MASK_FORMER.GATE_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        
        weight_dict.update({"loss_ce_gates": loss_ce_gates})

        losses = ["labels", "masks"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "gating_files": [os.path.join(cfg.MODEL.GATING.TRAIN_FOLDER, f"{f:s}.json") for f in cfg.MODEL.GATING.TRAIN_FILES],
            # "gating_train_threshold": cfg.MODEL.GATING.TRAIN_THRESHOLD,
            "gating_train_coeff": cfg.MODEL.GATING.TRAIN_COEFF,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "gating_output_file": os.path.join(cfg.OUTPUT_DIR, f"gates_{'_'.join([cfg.MODEL.WEIGHTS.split('/')[-2], ''.join(filter(str.isdigit,cfg.MODEL.WEIGHTS.split('/')[-1]))])}.csv"), # TO-DELETE
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        outputs, gates = self.sem_seg_head(features)

        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)

                # # TODO
                ############# Cityscapes #############
                # # hard version
                # gating_targets = F.one_hot(torch.LongTensor(
                #     [self.gating_targets[x["image_id"]] for x in batched_inputs]), 
                #     num_classes=4).to(torch.float).to(self.device) # Cityscapes; num_classes=5 for l2~6
                # # soft version
                # gating_targets = torch.from_numpy(
                #     np.array([self.gating_targets[x['image_id'].split('/')[-1].replace('.jpg', '.png')] for x in batched_inputs])).to(self.device)
                
                ############# COCO #############
                # # hard version
                # gating_targets = F.one_hot(torch.LongTensor(
                #     [self.gating_targets[x['file_name'].split('/')[-1].replace('.jpg', '.png')] for x in batched_inputs]), 
                #     num_classes=5).to(torch.float).to(self.device) # COCO; original=5
                # # soft version
                gating_targets = F.softmax(torch.from_numpy(
                    np.array([self.gating_targets[x['file_name'].split('/')[-1].replace('.jpg', '.png')] for x in batched_inputs])), dim=0).to(self.device)
            else:
                targets = None
                gating_targets = None

            # bipartite matching-based loss
            losses = self.criterion(outputs, gates, targets, gating_targets)

            # index of layer passed before exiting pixel decoder
            i_exit = self.sem_seg_head.pixel_decoder.transformer.encoder.i_exit + 2 # exit at layer 2 when i_exit = 0

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    # losses[k] = losses[k] * self.criterion.weight_dict[k] + i_exit
                    losses[k] *= (self.criterion.weight_dict[k] * i_exit) # linear combination of losses at each layer
                    # losses[k] *= (self.criterion.weight_dict[k]*i_exit*i_exit/10.0)
                    # losses[k] *= self.criterion.weight_dict[k] # original loss
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r
                
                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["instances"] = instance_r

            # # writing to csv file
            # with open(self.gating_output_file, 'a') as csvfile:
            #     writer = csv.writer(csvfile, delimiter=',')
            #     writer.writerow([batched_inputs[0]["image_id"], *gates[0].tolist()])
            # print('exit @', self.sem_seg_head.pixel_decoder.transformer.encoder.i_exit+2)

            return processed_results
        
    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets
    
    def prepare_gating_targets(self, files, slop):
        gating_targets = {}
        data = {}
        for path in files:
            f = open(path)
            d = json.load(f)
            
            for img in d.keys():
                if data.get(img):
                    data[img].append(d[img]["All"]['pq'])
                else:
                    data[img] = [d[img]["All"]['pq']]
        
        # # higher than a threshold for the first time
        # n_cls = len(files)
        # gating_targets = {k.replace("_gtFine_panoptic.png", ""): next((i for i, j in enumerate(v) if j>threshold), n_cls-1) 
        #                   for k,v in data.items()}

        # # highest PQ among all layers    
        # gating_targets = {k.replace("_gtFine_panoptic.png", ""): np.argmax(v) 
        #                   for k,v in data.items()} 

        # gflops = np.array([411.981, 443.448, 474.914, 506.381, 537.848]) # gflops at each layer

        # _slop = - slop
        # on Cityscapes: 
        # _slop = 0:    # counts: [118, 225, 426, 760, 1446]
        # _slop = -0.001 # counts: [151, 304, 518, 820, 1182]
        # _slop = -0.002 # counts: [198, 407, 618, 808, 944]
        # _slop = -0.003 # counts: [249, 510, 686, 785, 745]
        # _slop = -0.005 # counts: [374, 707, 735, 670, 489]
        # _slop = -0.01 # counts: [780, 928, 642, 390, 235]
        # _slop = -0.02 # counts: [1472, 820, 417, 178, 88]
        # _slop = -0.05 # counts: [2353, 510, 97, 13, 2]
        # _slop = -0.1 # counts: [2894, 79, 2, 0, 0]
                    
        # # use (PQ - i^2 * _slop)
        # _slop = 0.001;    # [313 814 906 636 306]
        # _slop = 0.0005;   # [210 512 804 853 596]
        
                    
        # on COCO: 
        # _slop = 0:    # counts: [21501, 20545, 20789, 22985, 32467]
        # _slop = -0.0005: #count: [27303 23991 22626 21509 22858]
        # _slop = -0.001 # counts: [32569 25847 22681 19535 17655]
        # _slop = -0.002 # counts: [40968 27148 21501 16287 12383]
        # _slop = -0.005 # counts: [56042 26533 17483 11148  7081]
        # _slop = -0.01 # counts: [68132 24135 13660  7861  4499]
        # _slop = -0.02 # counts: [80559 20637  9867  4917  2307]
        # _slop = -0.05 # counts: [98129 14316  4215  1278   349]
        # _slop = -0.1 # counts: [110327, 6730, 989, 203, 38]
                    
        # # use (PQ - i^2 * _slop)
        # _slop = 0.001;   # [50063 32479 19591 10573  5581]
        # _slop = 0.0005;   # [39677 32267 23396 14509  8438]

        # _, c = np.unique(list(gating_targets.values()), return_counts=True)
                    
        # TODO: previous method in main paper
        # gating_targets = {k.replace("_gtFine_panoptic.png", ""): np.argmax(np.array(v) - np.arange(1,6)*slop) # original: np.arange(1,6)
        #                   for k,v in data.items()}
        # soft version
        gating_targets = {k.replace("_gtFine_panoptic.png", ""): np.array(v) - np.arange(1,6)*slop # original: np.arange(1,6)
                          for k,v in data.items()}
        
        # # debug start
        # from IPython import embed; embed()
        # a = {k.replace("_gtFine_panoptic.png", ""): np.argmax(np.array(v) - np.arange(1,4)*0.0) # original: np.arange(1,6)
        #                   for k,v in data.items()}
        # _, c_a = np.unique(list(a.values()), return_counts=True)
        # # debug end 

        return gating_targets
    

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.sem_seg_head.num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result
