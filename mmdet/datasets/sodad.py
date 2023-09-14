import itertools
import logging
import os.path as osp
import tempfile
import time

import torch
import mmcv
import numpy as np
from mmcv.utils import print_log
from pycocotools.coco import COCO
from .sodad_eval.sodadeval import SODADeval
from terminaltables import AsciiTable
from collections import defaultdict
from functools import partial
from multiprocessing import Pool

from mmdet.core import eval_recalls
from .builder import DATASETS
from .custom import CustomDataset
from mmcv.ops.nms import nms


@DATASETS.register_module()
class SODADDataset(CustomDataset):
    CLASSES = ('people', 'rider', 'bicycle', 'motor', 'vehicle',
               'traffic-sign', 'traffic-light', 'traffic-camera',
               'warning-cone')  # only foreground categories available

    def __init__(self,
                 ori_ann_file,
                 **kwargs):
        super(SODADDataset, self).__init__(**kwargs)
        # self.ori_infos = self.load_ori_annotations(ori_ann_file)
        self.ori_coco = COCO(ori_ann_file)
        self.ori_img_ids = self.ori_coco.getImgIds()

    def load_ori_annotations(self, ori_ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.ori_coco = COCO(ori_ann_file)
        cat_ids = self.ori_coco.getCatIds(catNms=self.CLASSES)
        cat2label = {cat_id: i for i, cat_id in enumerate(cat_ids)}
        self.ori_img_ids = self.ori_coco.getImgIds()
        ori_infos = []
        for i in self.ori_img_ids:
            info = self.ori_coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            ori_infos.append(info)
        return ori_infos

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds(catNms=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.getImgIds()
        data_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
        return data_infos

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return [ann['category_id'] for ann in ann_info]

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.data_infos):
            if self.filter_empty_gt and self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def get_subset_by_classes(self):
        """Get img ids that contain any category in class_ids.

        Different from the coco.getImgIds(), this function returns the id if
        the img contains one of the categories rather than all.

        Args:
            class_ids (list[int]): list of category ids

        Return:
            ids (list[int]): integer list of img ids
        """

        ids = set()
        for i, class_id in enumerate(self.cat_ids):
            ids |= set(self.coco.cat_img_map[class_id])
        self.img_ids = list(ids)

        data_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
        return data_infos

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        # seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            # seg_map=seg_map
        )

        return ann

    # def _get_ori_img_ids(self):
    #     ids = set()
    #     for i, class_id in enumerate(self.cat_ids):
    #         ids |= set(self.ori_coco.cat_img_map[class_id])
    #     self.ori_img_ids = list(ids)

    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def _proposal2json(self, results):
        """Convert proposal results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            bboxes = results[idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self.ori_img_ids)):
            img_id = self.ori_img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results

    def _segm2json(self, results):
        """Convert instance segmentation results to COCO json style."""
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = self.cat_ids[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
        gt_bboxes = []
        for i in range(len(self.img_ids)):
            ann_ids = self.coco.getAnnIds(imgIds=self.img_ids[i])
            ann_info = self.coco.loadAnns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        # assert isinstance(results, list), 'results must be a list'
        # assert len(results) == len(self), (
        #     'The length of results is not equal to the dataset len: {} != {}'.
        #     format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir

    def translate(self, bboxes, x, y):
        dim = bboxes.shape[-1]
        translated = bboxes + \
                     np.array([x, y] * int(dim / 2), dtype=np.float32)
        return translated

    def merge_dets(self,
                           results,
                           with_merge=True,
                           nms_iou_thr=0.5,
                           nproc=10,
                           save_dir=None,
                           **kwargs):
        if mmcv.is_list_of(results, tuple):
            dets, segms = results
        else:
            dets = results

        # get patch results for evaluating
        if not with_merge:
            results = [(data_info['id'], result)
                       for data_info, result in zip(self.data_infos, results)]
            # TODO:
            if save_dir is not None:
                pass
            return results

        print('\n>>> Merge detected results of patch for whole image evaluating...')
        start_time = time.time()
        collector = defaultdict(list)
        # ensure data_infos and dets have the same length
        for data_info, result in zip(self.data_infos, dets):
            x_start, y_start = data_info['start_coord']
            new_result = []
            for i, res in enumerate(result):
                bboxes, scores = res[:, :-1], res[:, [-1]]
                bboxes = self.translate(bboxes, x_start, y_start)
                labels = np.zeros((bboxes.shape[0], 1)) + i
                new_result.append(np.concatenate(
                    [labels, bboxes, scores], axis=1
                ))

            new_result = np.concatenate(new_result, axis=0)
            collector[data_info['ori_id']].append(new_result)

        merge_func = partial(_merge_func, CLASSES=self.CLASSES, iou_thr=nms_iou_thr)
        if nproc > 1:
            pool = Pool(nproc)
            merged_results = pool.map(merge_func, list(collector.items()))
            pool.close()
        else:
            merged_results = list(map(merge_func, list(collector.items())))

        # TODO:
        if save_dir is not None:
            pass

        stop_time = time.time()
        print('Merge results completed, it costs %.1f seconds.' % (stop_time - start_time))
        return merged_results

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None,
                 with_merge=True):
        """Evaluation in COCO protocol.
        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        merged_results = self.merge_dets(
            results=results,
            with_merge=with_merge,
            nms_iou_thr=0.5,
            nproc=8,
            save_dir=None
        )

        img_ids = [result[0] for result in merged_results]
        results = [result[1] for result in merged_results]

        # sort the detection results based on the original annotation order which
        # conform to COCO API.
        empty_result = [np.empty((0, 5), dtype=np.float32) for cls in self.CLASSES]
        sort_results = []
        for ori_img_id in self.ori_img_ids:
            if ori_img_id in img_ids:
                sort_results.append(results[img_ids.index(ori_img_id)])
            else:
                sort_results.append(empty_result)
        results = sort_results

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        """ Evaluating detection results on top of COCO API """
        eval_results = {}
        cocoGt = self.ori_coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                cocoDt = cocoGt.loadRes(result_files[metric])
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            iou_type = 'bbox' if metric == 'proposal' else metric
            SODADEval = SODADeval(cocoGt, cocoDt, iou_type)
            SODADEval.params.catIds = self.cat_ids  # self.cat_ids
            SODADEval.params.imgIds = self.ori_img_ids
            SODADEval.params.maxDets = list(proposal_nums)
            SODADEval.params.iouThrs = iou_thrs
            # mapping of cocoEval.stats
            tod_metric_names = {
                'AP': 0,
                'AP_50': 1,
                'AP_75': 2,
                'AP_eS': 3,
                'AP_rS': 4,
                'AP_gS': 5,
                'AP_Normal': 6,
                'AR@100': 7,
                'AR@300': 8,
                'AR@1000': 9,
                'AR_eS@1000': 10,
                'AR_rS@1000': 11,
                'AR_gS@1000': 12,
                'AR_Normal@1000': 13
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in tod_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            if metric == 'proposal':
                SODADEval.params.useCats = 0
                SODADEval.evaluate()
                SODADEval.accumulate()
                SODADEval.summarize()
                if metric_items is None:
                    metric_items = [
                        'AP', 'AP_50', 'AP_75', 'AP_eS',
                        'AP_rS', 'AP_gS', 'AP_Normal'
                    ]

                for item in metric_items:
                    val = float(
                        f'{SODADEval.stats[tod_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                SODADEval.evaluate()
                SODADEval.accumulate()
                SODADEval.summarize()

                if True:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = SODADEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.ori_coco.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)

                if metric_items is None:
                    metric_items = [
                        'AP', 'AP_50', 'AP_75', 'AP_eS',
                        'AP_rS', 'AP_gS', 'AP_Normal'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = float(
                        f'{SODADEval.stats[tod_metric_names[metric_item]]:.3f}'
                    )
                    eval_results[key] = val
                ap = SODADEval.stats[:7]
                eval_results[f'{metric}_mAP_copypaste'] = (
                    f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} '
                    f'{ap[3]:.3f} {ap[4]:.3f} {ap[5]:.3f} '
                    f'{ap[6]:.3f} '
                )

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results

def _merge_func(info, CLASSES, iou_thr):
    img_id, label_dets = info
    label_dets = np.concatenate(label_dets, axis=0)
    labels, dets = label_dets[:, 0], label_dets[:, 1:]

    ori_img_results = []
    for i in range(len(CLASSES)):
        cls_dets = dets[labels == i]
        if len(cls_dets) == 0:
            ori_img_results.append(np.empty((0, dets.shape[1]), dtype=np.float32))
            continue
        bboxes, scores = cls_dets[:, :-1], cls_dets[:, [-1]]
        bboxes = torch.from_numpy(bboxes).to(torch.float32).contiguous()
        scores = torch.from_numpy(np.squeeze(scores, 1)).to(torch.float32).contiguous()
        results, inds = nms(bboxes, scores, iou_thr)
        # If scores.shape=(N, 1) instead of (N, ), then the results after NMS
        # will be wrong. Suppose bboxes.shape=[N, 4], the results has shape
        # of [N**2, 5],
        results = results.numpy()
        ori_img_results.append(results)
    return img_id, ori_img_results



