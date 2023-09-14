#==============================================================================================
#   Note:
#       This project depends on the third-library: `shapely`, `pycocotools`
#==============================================================================================

import os
import os.path as osp
import random
from tqdm import tqdm
from time import time
import argparse
import logging
import shutil
import copy

import cv2
import numpy as np
import json
import shapely.geometry as shgeo
from pycocotools.coco import COCO
from multiprocessing import Manager, Pool
from functools import partial, reduce

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import warnings
warnings.filterwarnings("ignore", category=Warning)

class SODADSplit(object):
    """
    Args:
        rootDir: (str) path to save original images and annotation file (json format).
        patchH: (int) height of divide image patch.
        patchW: (int) width of divide image patch.
        stepSzH: (int) step size in height-axis while dividing the image.
        stepSzW: (int) step size in width-axis while dividing the image.
        interAreaIgnThr: (float) Set ignore to those regions whose area ratio below the threshold.
        ignFillVal: (tuple[int]) value to fill the ignore regions, default to (0, 0, 0). Note it
            follows BGR order.
    Returns:
        None
    """

    def __init__(
            self,
            nproc,
            mode,
            oriImgDir,
            oriAnnDir,
            patchH,
            patchW,
            stepSzH,
            stepSzW,
            padVal=(0, 0, 0),
            ignFillVal=(0, 0, 0),
            interAreaIgnThr=0,
            splDir=None,
            isVis=False
        ):
        self.nproc = nproc
        self.mode = mode
        self.oriImgDir = oriImgDir
        self.oriAnnDir = oriAnnDir
        self.divImgDir = osp.join(splDir, 'Images', self.mode)
        self.divAnnDir = osp.join(splDir, 'Annotations')
        self._rmdir(self.divImgDir)
        self._mkdir(self.divImgDir)
        self._mkdir(self.divAnnDir)

        self.patchH = patchH
        self.patchW = patchW
        self.stepSzH = stepSzH
        self.stepSzW = stepSzW
        self.padVal = padVal
        self.interAreaIgnThr = interAreaIgnThr
        self.ignFill = True if ignFillVal else False
        if self.ignFill:
            self.ignFillVal = ignFillVal
        self.annPth = osp.join(self.oriAnnDir, self.mode + '.json')
        self.divAnnPth = osp.join(self.divAnnDir, self.mode + '.json')
        self.isVis = isVis
        if self.isVis:
            self.visDir = osp.join(splDir, 'Vis')
            self._rmdir(self.visDir)
            self._mkdir(self.visDir)

        self.newImgs = dict()
        self.newImgToAnns = dict()
        self.newImgId = 0
        self.newInsId = 0
        self.count = 0

        # prepare annotations
        self._getIndexLst()

    def _rmdir(self, pth):
        if osp.exists(pth):
            shutil.rmtree(pth)

    def _mkdir(self, pth):
        if osp.exists(pth):
            pass
        else:
            if osp.exists(osp.dirname(pth)):
                os.mkdir(pth)
            else:
                os.mkdir(osp.dirname(pth))
                os.mkdir(pth)


    def _loadAnn(self):
        " load annotation info with the help of pycocotools "
        self.annInfo = COCO(self.annPth)

    def _getAnnandImg(self):
        """
        :return imgToAnns: annotation info with image_id indexed
        :return imgs: images info with image_id indexed
        """
        self._loadAnn()
        self.imgToAnns = self.annInfo.imgToAnns
        self.imgs = self.annInfo.imgs
        assert len(self.imgToAnns) <= len(self.imgs), \
            logger.error(
                f'number of imgToAnns ({len(self.imgToAnns)}) must <= '
                f'number of imgs ({len(self.imgs)})')

    def _xywh2poly(self, bbox):
        """ Convert bbox annotation from xywh to polygon format, whose order follows:
            top-left -> top-right -> bottom-right -> bottom-left """
        polyLst = [(bbox[:2]),
                   (bbox[0] + bbox[2], bbox[1]),
                   (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                   (bbox[0], bbox[1] + bbox[3])]
        poly = shgeo.Polygon(polyLst)
        polyLst = list(np.array(polyLst).ravel())
        polyLst = [int(poly) for poly in polyLst]
        return poly, polyLst

    def _addPoly2Anns(self, anns):
        """ 1. Add poly-info to annotations;
            2. Integrate ann-info to construct a new sequential annotation dict whose keys includes:
                `bbox`, `image_id`, `category_id`, `area`, `ignore`, `id`, `poly` """
        annsWithPoly = dict()
        for key, value in anns[0].items():
            annsWithPoly[key] = []
        annsWithPoly['poly'] = []
        for ann in anns:
            for key, value in ann.items():
                annsWithPoly[key].append(value)
                if key == 'bbox':
                    poly, _ = self._xywh2poly(bbox=value)
                    annsWithPoly['poly'].append(poly)
        return annsWithPoly

    def _getIndexLst(self):
        self._getAnnandImg()
        self.imgIds = []
        for key, value in self.imgToAnns.items():
            self.imgIds.append(key)

    def _pad(self, inImg):
        h, w, c = inImg.shape
        padImg = np.full((max(h, 800), max(w, 800), c), self.ignFillVal)
        padImg[:h, :w, :] = inImg
        return padImg

    def _divImg(self, img, fileName, imgH, imgW, imgExAnn, annsWithPoly, imgId):
        """
        newFileName = fileName + f'_{curPatch}'
        """
        newImgToAnns = []
        newImgs = []
        xmin, ymin, xmax, ymax = imgExAnn
        # avoid duplicated patches after dividing during training phase
        if self.mode == 'train':
            imgW, imgH = min(imgW, xmax+5), min(imgH, ymax+5)
        curPatch = 0
        # pad image
        if imgW < 800 or imgH < 800:
            img = self._pad(img)
            imgW, imgH = max(imgW, 800), max(imgH, 800)
        left, up = 0, 0
        while (left < imgW):
            if (left + self.patchW >= imgW):
                left = max(imgW - self.patchW, 0)
            up = 0
            while (up < imgH):
                if (up + self.patchH >= imgH):
                    up = max(imgH - self.patchH, 0)
                right = left + self.patchW
                down = up + self.patchH
                # save images
                imgPatch = img[up: down, left: right, :]
                newFileName = fileName + f'_{curPatch}.jpg'

                # collect patch annotations and images
                patchBbox = [left, up, right-left, down-up]
                patchAnns, fillIgnImgPatch = self._divAnn(
                    annsWithPoly, patchBbox, self.newImgId, imgPatch)

                # random visualize the images if `isVis`
                if self.isVis:
                    visProb = random.random()
                    if visProb > 0.99:
                        self.rdmVis(patchAnns, fillIgnImgPatch, newFileName)

                # save image patch with valuable annotations only during training
                sign = len(patchAnns) != 0 if self.mode == 'train' else True
                if sign:
                    newImgs.append(dict(
                        file_name=newFileName,
                        height= down - up,
                        width=right - left,
                        id=self.newImgId,
                        ori_id=imgId,
                        start_coord=(left, up))
                    )
                    # save original image when no ignore-fill process conducted
                    if self.ignFill:
                        cv2.imwrite(osp.join(self.divImgDir, newFileName), fillIgnImgPatch)
                    else:
                        cv2.imwrite(osp.join(self.divImgDir, newFileName), imgPatch)
                    # collect divided ann
                    newImgToAnns.append(patchAnns)

                    # update the generated image id
                    self.newImgId += 1

                    # update patch id
                    curPatch += 1
                    self.count += 1

                if (up + self.patchH >= imgH):
                    break
                else:
                    up = up + self.patchH - self.stepSzH
            if (left + self.patchW >= imgW):
                break
            else:
                left = left + self.patchW - self.stepSzW
        return newImgToAnns, newImgs

    def _divAnn(self, annsWithPoly, patchBbox, imgId, imgPatch):
        [left, up] = patchBbox[:2]
        polys = annsWithPoly['poly']
        catIds = annsWithPoly['category_id']
        oriAreas = [poly.area for poly in polys]
        patchPoly, _ = self._xywh2poly(patchBbox)
        intersTmp = [patchPoly.intersection(poly) for poly in polys]
        inters = []
        bounds = []
        for i, inter in enumerate(intersTmp):
            if inter.area > 0:
                bounds.append(list(inter.bounds))
                inters.append(inter)
            else:
                bounds.append([])
                inters.append(0)
        # filter those no-intersection boxes
        vldIndex = list()
        for i, bound in enumerate(bounds):
            if len(bound) != 0:
                vldIndex.append(i)
        catIds = list(np.array(catIds)[vldIndex])
        oriAreas = list(np.array(oriAreas)[vldIndex])
        inters = list(np.array(inters)[vldIndex])
        bounds = list(np.array(bounds)[vldIndex])

        fillIgnImgPatch = None
        if self.ignFill:
            fillIgnImgPatch = self._fillIgn(imgPatch, catIds, bounds, left, up) # if bounds else None
            # filter ignore annotations when fill the ignores regions
            vldSign = np.array(catIds) != 10
            catIds = list(np.array(catIds)[vldSign])
            oriAreas = list(np.array(oriAreas)[vldSign])
            try:
                inters = list(np.array(inters)[vldSign])
            except TypeError:
                raise TypeError
            bounds = list(np.array(bounds)[vldSign])

        patchAnns = []
        for i, bbox in enumerate(bounds):
            ann = dict()
            bbox = [bbox[0]-left, bbox[1]-up, bbox[2]-bbox[0], bbox[3]-bbox[1]] # (x, y, w, h)
            _, polyLst = self._xywh2poly(bbox)
            ann['segmentation'] = [polyLst]
            ann['bbox'] = bbox
            ann['category_id'] = int(catIds[i])
            ann['area'] = inters[i].area
            ann['image_id'] = imgId
            ann['id'] = self.newInsId
            # required for COCO evaluation, `True` when the object patch is too small
            ann['ignore'] = bool((inters[i].area / oriAreas[i]) <= self.interAreaIgnThr) \
                if ann['category_id'] != 10 else True
            ann['iscrowd'] = 0
            patchAnns.append(ann)

            # update instance id for each bbox
            self.newInsId += 1
        return patchAnns, fillIgnImgPatch

    def _fillIgn(self, imgPatch, catLst, bounds, left, up):
        maskSz = imgPatch.shape[:2]
        ignMask = self._getIgnMask(maskSz, catLst, bounds, left, up)
        ignB, ignG, ignR = self.ignFillVal  # BGR-format
        imgPatchB, imgPatchG, imgPatchR = cv2.split(imgPatch)
        imgPatchB = np.where(ignMask == 1, ignB, imgPatchB)
        imgPatchG = np.where(ignMask == 1, ignG, imgPatchG)
        imgPatchR = np.where(ignMask == 1, ignR, imgPatchR)
        fillIgnImgPatch = cv2.merge([imgPatchB, imgPatchG, imgPatchR])
        return fillIgnImgPatch

    def _getIgnMask(self, maskSz, catLst, bounds, left, up):
        """ Args:
                - maskSz: [height, width]
                - catLst: [1, 2, 5, 3, 9, ...]
                - bounds: see func: `_divAnn()`
            Return:
                - ignMask: 1 denotes regions to be filled """
        bboxLst = list()
        for bound in bounds:
            if len(bound) == 0:
                bboxLst.append([])
            else:
                bboxLst.append([bound[0]-left, bound[1]-up, bound[2]-left, bound[3]-up])
        ignMask = np.zeros(maskSz).astype(np.uint8)
        ignSign = (np.array(catLst) == 10)  # .reshape(-1, 1)  # category id of `ignore` is 10
        objBboxAry = np.array(bboxLst)[~ignSign].astype(np.int)
        ignBboxAry = np.array(bboxLst)[ignSign].astype(np.int)

        # assign 1 to ignore regions
        for ignBox in ignBboxAry:
            xmin, ymin, xmax, ymax = ignBox
            ignMask[ymin: ymax, xmin: xmax] = 1

        # re-assign 0 to those foreground regions which inside the ignore ones
        for objBox in objBboxAry:
            xmin, ymin, xmax, ymax = objBox
            ignMask[ymin: ymax, xmin: xmax] = 0
        return ignMask

    def divImgandAnn(self, imgId):
        # prepare image info
        imgInfo = self.imgs[imgId]
        fileName = imgInfo['file_name']    # use `file_name` in accordance to COCO format
        imgPth = osp.join(self.oriImgDir, fileName)
        fileName = fileName.split(os.sep)[-1].split('.')[0]
        img = cv2.imread(imgPth)
        imgH, imgW = imgInfo['height'], imgInfo['width']

        # prepare annotation info
        anns = self.imgToAnns[imgId]
        imgExAnn = self._getImgExAnn(anns)
        annsWithPoly = self._addPoly2Anns(anns)

        # divide image and corresponding annotation file
        newImgToAnns, newImgs = self._divImg(img, fileName, imgH, imgW, imgExAnn, annsWithPoly, imgId)
        return [newImgToAnns, newImgs]


    def _getImgExAnn(self, anns):
        xMin, yMin, xMax, yMax = 10000, 10000, 0, 0
        for ann in anns:
            bbox = [int(b) for b in ann['bbox']]
            xmin, ymin, width, height = bbox
            xMin = min(xMin, xmin)
            yMin = max(yMin, ymin)
            xMax = max(xMax, xmin+width)
            yMax = max(yMax, ymin+height)
        return [xMin, yMin, xMax, yMax]

    def rdmVis(self, patchAnns, fillIgnImgPatch, imgName):
        img = copy.deepcopy(fillIgnImgPatch)
        # random sample divided image patch for visualization
        COLORS = [(63, 38, 0), (252, 221, 115), (193, 17, 179), (184, 0, 246),
                  (127, 255, 1), (57, 125, 255), (247, 94, 132), (215, 242, 150),
                  (75, 13, 135), (65, 47, 255)
        ]
        for ann in patchAnns:
            bbox, catId = ann['bbox'], ann['category_id']
            xmin, ymin, w, h = [int(elem) for elem in bbox]
            cv2.rectangle(img, (xmin, ymin), (xmin + w, ymin + h), COLORS[catId - 1], 2)
        cv2.imwrite(osp.join(self.visDir, imgName), img)

    def mergeJson(self, newImgToAnns, newImgs):
        # type
        dataset_type = 'instance'
        # categories
        categories = [
            dict(id=1, name='people'),
            dict(id=2, name='rider'),
            dict(id=3, name='bicycle'),
            dict(id=4, name='motor'),
            dict(id=5, name='vehicle'),
            dict(id=6, name='traffic-sign'),
            dict(id=7, name='traffic-light'),
            dict(id=8, name='traffic-camera'),
            dict(id=9, name='warning-cone')
        ]
        # add `ignore` category if no fill-process performed
        if not self.ignFill:
            categories.append(
                dict(id=10, name='ignore')
            )
        # annotations and images
        annotations = []
        images = []
        newImgid = 0
        newInsid = 0
        for i in range(len(newImgToAnns)):
            if len(newImgToAnns[i]) != 0:
                for j in range(len(newImgToAnns[i])):
                    newImgToAnns[i][j]['id'] = newInsid
                    newImgToAnns[i][j]['image_id'] = newImgid
                    newInsid += 1
            newImgs[i]['id'] = newImgid
            annotations.extend(newImgToAnns[i])
            images.append(newImgs[i])
            newImgid += 1
        jsonRes = dict(
            type=dataset_type,
            images=images,
            annotations=annotations,
            categories=categories
        )
        return jsonRes

    def split(self):
        logger.info('Divide image and corresponding annotation file...\n')
        startTime = time()
        pool = Pool(self.nproc)
        outputs = pool.map(self.divImgandAnn, self.imgIds)
        pool.close()
        newImgToAnns, newImgs = [], []
        for output in outputs:
            newImgToAnns = newImgToAnns + output[0]
            newImgs = newImgs + output[1]
        jsonRes = self.mergeJson(newImgToAnns, newImgs)
        json.dump(jsonRes, open(self.divAnnPth, "w"), indent=4)
        stopTime = time()
        logger.info(f'\nFinished with {str(stopTime - startTime)} seconds totally.')

def add_parser(parser):
    """Add arguments."""
    parser.add_argument('--cfgJson', default="./split_configs/split_train.json",
                        help='config json for split images')
    parser.add_argument(
        '--mode', type=str, default='train')
    parser.add_argument(
        '--nproc', type=int, default=10, help='the procession number')

    # argument for loading data
    parser.add_argument(
        '--oriImgDir',
        nargs='+',
        type=str,
        default=None,
        help='dirs for original images')
    parser.add_argument(
        '--oriAnnDir',
        nargs='+',
        type=str,
        default=None,
        help='dirs for original annotations')

    # argument for splitting image
    parser.add_argument(
        '--patchH',
        nargs='+',
        type=int,
        default=800,
        help='the height of sliding windows')
    parser.add_argument(
        '--patchW',
        nargs='+',
        type=int,
        default=800,
        help='the width of sliding windows')
    parser.add_argument(
        '--stepSzH',
        nargs='+',
        type=int,
        default=150,
        help='the height of step size')
    parser.add_argument(
        '--stepSzW',
        nargs='+',
        type=int,
        default=150,
        help='the width of step size')

    parser.add_argument(
        '--padVal',
        type=int,
        default=[0, 0, 0],
        help='padding value for original images')
    parser.add_argument(
        '--ignFillVal',
        type=int,
        default=[0, 0, 0],
        help='ignore filling value for split images')
    parser.add_argument(
        '--interAreaIgnThr',
        type=float,
        default=0.40,
        help='the regions whose area ratios below the threshold will be ignored')

    # argument for saving
    parser.add_argument(
        '--splDir',
        type=str,
        default='.',
        help='dirs to save split images and annotations')

    parser.add_argument(
        '--isVis',
        action='store_true',    # default to False
        help='whether to save visualization after split')

def parse_args():
    parser = argparse.ArgumentParser(description='Split images and annotations of SODA-D.')
    add_parser(parser)
    args = parser.parse_args()

    # load split config
    if args.cfgJson is not None:
        with open(args.cfgJson, 'r') as f:
            prior_config = json.load(f)

        for action in parser._actions:
            if action.dest not in prior_config or \
                    not hasattr(action, 'default'):
                continue
            action.default = prior_config[action.dest]
        args = parser.parse_args()

    assert args.interAreaIgnThr >= 0 and args.interAreaIgnThr < 1
    if not osp.exists(args.splDir):
        os.mkdir(args.splDir)
    return args

def main():
    logger.info('Start...')
    args = parse_args()
    SS = SODADSplit(
        nproc=args.nproc,
        mode=args.mode,
        oriImgDir=args.oriImgDir,
        oriAnnDir=args.oriAnnDir,
        patchH=args.patchH,
        patchW=args.patchW,
        stepSzH=args.stepSzH,
        stepSzW=args.stepSzW,
        padVal=args.padVal,
        ignFillVal=args.ignFillVal,
        interAreaIgnThr=args.interAreaIgnThr,
        splDir=args.splDir,
        isVis=args.isVis
    )
    SS.split()

if __name__ == "__main__":
    main()
