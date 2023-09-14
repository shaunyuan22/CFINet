# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner
import numpy as np


@BBOX_ASSIGNERS.register_module()
class DynamicAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, or a semi-positive integer
    indicating the ground truth index.

    - -1: negative sample, no assigned gt
    - semi-positive integer: positive sample, index (0-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
            `min_pos_iou` is set to avoid assigning bboxes that have extremely
            small iou with GT as positive samples. It brings about 0.3 mAP
            improvements in 1x schedule but does not affect the performance of
            3x schedule. More comparisons can be found in
            `PR #7464 <https://github.com/open-mmlab/mmdetection/pull/7464>`_.
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
        match_low_quality (bool): Whether to allow low quality matches. This is
            usually allowed for RPN and single stage detectors, but not allowed
            in the second stage. Details are demonstrated in Step 4.
        gpu_assign_thr (int): The upper bound of the number of GT for GPU
            assign. When the number of gt is above this threshold, will assign
            on CPU device. Negative values mean not assign on CPU.
    """

    def __init__(self,
                 low_quality_iou_thr=0.2,
                 base_pos_iou_thr=0.25,
                 normal_iou_thr=0.70,
                 r=0.15,
                 base_size=12,
                 scale_ratio=1, 
                 neg_iou_thr=0.2,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 gpu_assign_thr=-1,
                 iou_calculator=dict(type='BboxOverlaps2D')):
        self.low_quality_iou_thr = low_quality_iou_thr
        self.base_pos_iou_thr = base_pos_iou_thr
        self.normal_iou_thr = normal_iou_thr
        self.r = r
        self.base_size = base_size
        self.scale_ratio = scale_ratio
        self.neg_iou_thr = neg_iou_thr
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.iou_calculator = build_iou_calculator(iou_calculator)

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None,
               num_base_anchors=1, scale_ratio=1.0):
        """Assign gt to bboxes.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, or a semi-positive number. -1 means negative
        sample, semi-positive number is the index (0-based) of assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every bbox to the background
        2. assign proposals whose iou with all gts < neg_iou_thr to 0
        3. for each bbox, if the iou with its nearest gt >= pos_iou_thr,
           assign it to that bbox
        4. for each gt bbox, assign its nearest proposals (may be more than
           one) to itself

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.

        """
        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
            gt_bboxes.shape[0] > self.gpu_assign_thr) else False
        # compute overlap and assign gt on CPU when number of GT is large
        if assign_on_cpu:
            device = bboxes.device
            bboxes = bboxes.cpu()
            gt_bboxes = gt_bboxes.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()
            if gt_labels is not None:
                gt_labels = gt_labels.cpu()

        overlaps = self.iou_calculator(gt_bboxes, bboxes, mode='iou')
        gt_areas = (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * \
                   (gt_bboxes[:, 2] - gt_bboxes[:, 0])

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            if self.ignore_wrt_candidates:
                ignore_overlaps = self.iou_calculator(
                    bboxes, gt_bboxes_ignore, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = self.iou_calculator(
                    gt_bboxes_ignore, bboxes, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        assign_result, ign_inds = \
            self.assign_wrt_overlaps(overlaps, gt_areas, gt_labels, num_base_anchors, scale_ratio=scale_ratio)
        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)
        return assign_result, ign_inds

    def assign_wrt_overlaps(self, overlaps, gt_areas, gt_labels=None, num_base_anchors=1, scale_ratio=1.0):
        """Assign w.r.t. the overlaps of bboxes with gts.

        Args:
            overlaps (Tensor): Overlaps between k gt_bboxes and n bboxes,
                shape(k, n).
            gt_labels (Tensor, optional): Labels of k gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        assigned_ign_inds = overlaps.new_full((num_bboxes,),
                                             0,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            assigned_ign_inds = overlaps.new_full((num_bboxes,),
                                                  0,
                                                  dtype=torch.long)

            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels), assigned_ign_inds

        potential_low_thr = overlaps.new_full(
            (num_bboxes, ), self.base_pos_iou_thr, dtype=overlaps.dtype)

        # choose the index of anchor with max_iou per location
        overlaps = overlaps.view(num_gts, num_base_anchors, -1)
        overlaps, max_inds_per_loc = overlaps.max(dim=1)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

        # 2. assign positive: anchors below the lower bound and
        # above the upper bound are set to be 1
        gt_pos_thrs = self.get_gt_pos_thrs(gt_areas, scale_ratio=scale_ratio)
        for i in range(num_gts):
            inds = (argmax_overlaps == i) & \
                   (max_overlaps >= gt_pos_thrs[i])
            potential_low_thr[inds] = gt_pos_thrs[i]

        pos_inds = max_overlaps >= potential_low_thr
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        # 3. assign negtive: below the lower bound
        neg_inds = max_overlaps < self.neg_iou_thr
        assigned_gt_inds[neg_inds] = 0

        # 4. assign ignore:
        ign_inds = assigned_gt_inds == -1
        assigned_ign_inds[ign_inds] = 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels), \
               assigned_ign_inds

    def get_area_id(self, areas, scale_ratio):
        # unused
        area_id = torch.full_like(areas, 3, dtype=torch.long)
        area_thrs = [0, 144, 400, 1024, 2000]  # for SODA dataset
        for i, area in enumerate(area_thrs):
            if i == 0:
                continue
            up_area = area * (scale_ratio ** 2)
            down_area = area_thrs[i-1] * (scale_ratio ** 2)
            inds = (areas > down_area) & (areas <= up_area)
            area_id[inds] = i - 1
        return area_id

    def get_gt_pos_thrs(self, areas, scale_ratio):
        areas = torch.sqrt(areas / (scale_ratio ** 2))
        thrs = torch.max(torch.tensor(self.low_quality_iou_thr).cuda(),
                         torch.tensor(self.base_pos_iou_thr).cuda() + self.r * torch.log2(areas / self.base_size))
        thrs = torch.min(torch.tensor(self.normal_iou_thr).cuda(), thrs)
        return thrs