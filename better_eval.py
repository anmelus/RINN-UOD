# https://github.com/OCR-D/ocrd_segment/blob/master/ocrd_segment/evaluate.py

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import numpy as np
from itertools import chain

def evaluate_coco(coco_gt, coco_dt, parameters, catIds=None):
    #LOG = getLogger('processor.EvaluateSegmentation')
    #LOG.info("comparing segmentations")
    stats = dict(parameters)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox') # bbox
    cocoGt = COCO('./modified/modified_val_data_gt.json')
    imgIds = cocoGt.getImgIds(catIds = 1)
    coco_eval.params.catIds = [1]
    coco_eval.params.imgIds = imgIds

    if catIds:
       coco_eval.params.catIds = catIds
    coco_eval.params.maxDets = [1, 100, 20000] # unlimited nr of detections (requires pycocotools#559) #TODO
    #       All non-zero pairs are considered matches if their intersection over union > 0.5 _or_
    #       their intersection over either side > 0.5. Matches can thus be n:m.
    #       Non-matches are counted as well (false positives and false negatives).
    #       Aggregation uses microaveraging over images. Besides counting segments,
    #       the pixel areas are counted and averaged (as ratios).
    coco_eval.evaluate()
    # get by-page alignment (ignoring inadequate 1:1 matching by pycocotools)
    def get(arg):
        return lambda x: x[arg]
    numImgs = len(coco_eval.params.imgIds)
    numAreas = len(coco_eval.params.areaRng)
    early_fp_counter=0
    for imgind, imgId in enumerate(coco_eval.params.imgIds):
        img = coco_gt.imgs[imgId]
        pageId = img['file_name']
        for catind, catId in enumerate(coco_eval.params.catIds):
            print(f"ind: {catind}, cat Id: {catId}")
            assert(catind < 1)
            cat = coco_gt.cats[catId]
            catName = cat['name']
            if not catId:
                assert(False)
                continue
            # bypassing COCOeval.evaluateImg, hook onto its results
            # (again, we stay at areaRng[0]=all and maxDets[0]=all)
            start = catind * numImgs * numAreas
            evalimg = coco_eval.evalImgs[start + imgind]
            if evalimg is None:
                continue # no DT and GT here
            # record as dict by pageId / by category
            imgstats = stats.setdefault('by-image', dict())
            pagestats = imgstats.setdefault(pageId, dict())
            # get matches and ious and scores
            ious = coco_eval.ious[imgId, catId]

            # TODO: Only takes bounding boxes with an IoU greater than 0.
            """ Example:
            For image 397133, there are 19 total ground truth bounding boxes. If our model predicts 81 boxes there are a total of 1539 pairwise iou calculations. 
            But only ~105 are used for calculations since they hit the ground truth bounding boxes
            
            Intuitively it makes sense; if a bounding box detects one object it shouldnt be classified as a false postive for the others.
            But we have dozens of bounding boxes detecting random objects and have an IoU=0 for all objects in a picture and since those objects are not in the coco dataset.

            This is why precision ends up being so high, we are throwing away bounding boxes that are not detecting anything in the dataset.
            """

                        #-------------
            if len(ious):
                for item in ious:
                    if (np.allclose(item, np.zeros(item.shape))):
                        early_fp_counter += 1
                overlaps_dt, overlaps_gt = ious.nonzero()
            else:
                overlaps_dt = overlaps_gt = []

            #--------------
            
            # reconstruct score sorting in computeIoU
            gt = coco_eval._gts[imgId, catId]
            dt = coco_eval._dts[imgId, catId]
            dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
            dt = [dt[i] for i in dtind]
            matches = list()
            gtmatches = dict()
            dtmatches = dict()
            for dtind, gtind in zip(overlaps_dt, overlaps_gt):
                d = dt[dtind]
                g = gt[gtind]
                iou = ious[dtind, gtind]
                # TODO
                gtcoords = {
                    'left': g['bbox'][0],
                    'right': g['bbox'][0]+g['bbox'][2],
                    'top': g['bbox'][1],
                    'bottom': g['bbox'][1]+g['bbox'][3]
                }
                predcoords = {
                    'left': d['bbox'][0],
                    'right': d['bbox'][0]+d['bbox'][2],
                    'top': d['bbox'][1],
                    'bottom': d['bbox'][1]+d['bbox'][3]
                }
                x_overlap = max(0, min(gtcoords['right'], predcoords['right']) - max(gtcoords['left'], predcoords['left']))
                y_overlap = max(0, min(gtcoords['bottom'], predcoords['bottom']) - max(gtcoords['top'], predcoords['top']))
                intersection = x_overlap * y_overlap
                union = g['bbox'][2]*g['bbox'][3]+d['bbox'][2]*d['bbox'][3] - intersection
                # print(intersection)
                #union = maskArea(mergeMasks([g['segmentation'], d['segmentation']]))
                intersection = int(iou * union)
                # print(intersection)
                # cannot use g or d['area'] here, because mask might be fractional (only-fg) instead of outline
                areag = int(g['bbox'][2]*g['bbox'][3])
                aread = int(d['bbox'][2]*d['bbox'][3])
                iogt = intersection / areag
                iodt = intersection / aread
                if iou < 0.5 and iogt < 0.5 and iodt < 0.5:
                    continue
                gtmatches.setdefault(gtind, list()).append(dtind)
                dtmatches.setdefault(dtind, list()).append(gtind)
                matches.append((g['id'],
                                d['id'],
                                iogt, iodt, iou, intersection))
                pagestats.setdefault('true_positives', dict()).setdefault(catName, list()).append(
                    {'GT.ID': g['image_id'],
                     'DT.ID': d['image_id'],
                     'GT.area': areag,
                     'DT.area': aread,
                     'I.area': intersection,
                     'IoGT': iogt,
                     'IoDT': iodt,
                     'IoU': iou})
            dtmisses = []
            for dtind, d in enumerate(dt):
                if dtind in dtmatches:
                    continue
                dtmisses.append((d['id'], aread))                                                  # TODO          #maskArea(d['bbox'])))
                pagestats.setdefault('false_positives', dict()).setdefault(catName, list()).append(
                    {'DT.ID': d['image_id'],
                     'area': int(d['area'])})
            gtmisses = []
            for gtind, g in enumerate(gt):
                if gtind in gtmatches:
                    continue
                gtmisses.append((g['id'], areag))                             # TODO maskArea(g['bbox'])))
                pagestats.setdefault('false_negatives', dict()).setdefault(catName, list()).append(
                    {'GT.ID': g['image_id'],
                     'area': int(g['area'])})
            # measure under/oversegmentation for this image and category
            # (follows Zhang et al 2021: Rethinking Semantic Segmentation Evaluation [arXiv:2101.08418])
            over_gt = set(gtind for gtind in gtmatches if len(gtmatches[gtind]) > 1)
            over_dt = set(chain.from_iterable(
                gtmatches[gtind] for gtind in gtmatches if len(gtmatches[gtind]) > 1))
            under_dt = set(dtind for dtind in dtmatches if len(dtmatches[dtind]) > 1)
            under_gt = set(chain.from_iterable(
                dtmatches[dtind] for dtind in dtmatches if len(dtmatches[dtind]) > 1))
            over_degree = sum(len(gtmatches[gtind]) - 1 for gtind in gtmatches)
            under_degree = sum(len(dtmatches[dtind]) - 1 for dtind in dtmatches)
            if len(dt) and len(gt):
                oversegmentation = len(over_gt) * len(over_dt) / len(gt) / len(dt)
                undersegmentation = len(under_gt) * len(under_dt) / len(gt) / len(dt)
                # Zhang's idea of attenuating the under/oversegmentation ratio with a "penalty"
                # to account for the degree of further sub-segmentation is misguided IMHO,
                # because its degree term depends on the total number of segments:
                # oversegmentation = np.tanh(oversegmentation * over_degree)
                # undersegmentation = np.tanh(undersegmentation * under_degree)
                pagestats.setdefault('oversegmentation', dict())[catName] = oversegmentation
                pagestats.setdefault('undersegmentation', dict())[catName] = undersegmentation
                pagestats.setdefault('precision', dict())[catName] =  (len(dt) - len(dtmisses)) / len(dt)
                pagestats.setdefault('recall', dict())[catName] =  (len(gt) - len(gtmisses)) / len(gt)
            tparea = sum(map(get(5), matches)) # sum(inter)
            fparea = sum(map(get(1), dtmisses)) # sum(area)
            fnarea = sum(map(get(1), gtmisses)) # sum(area)
            if tparea or (fparea and fnarea):
                pagestats.setdefault('pixel_precision', dict())[catName] = tparea / (tparea + fparea)
                pagestats.setdefault('pixel_recall', dict())[catName] =  tparea / (tparea + fnarea)
                pagestats.setdefault('pixel_iou', dict())[catName] =  tparea / (tparea + fparea + fnarea)
            # aggregate per-img/per-cat IoUs for microaveraging
            evalimg['matches'] = matches # TP
            evalimg['dtMisses'] = dtmisses # FP
            evalimg['gtMisses'] = gtmisses # FN
            evalimg['dtIdsOver'] = [dt[dtind]['id'] for dtind in over_dt]
            evalimg['gtIdsOver'] = [gt[gtind]['id'] for gtind in over_gt]
            evalimg['dtIdsUnder'] = [dt[dtind]['id'] for dtind in under_dt]
            evalimg['gtIdsUnder'] = [gt[gtind]['id'] for gtind in under_gt]

    catstats = stats.setdefault('by-category', dict())
    # accumulate our over-/undersegmentation and IoU ratios
    numImgs = len(coco_eval.params.imgIds)
    numAreas = len(coco_eval.params.areaRng)
    for catind, catId in enumerate(coco_eval.params.catIds):
        cat = coco_gt.cats[catId]
        catstats.setdefault(cat['name'], dict())
        start = catind * numImgs * numAreas
        # again, we stay at areaRng[0]=all and maxDets[0]=all
        evalimgs = [coco_eval.evalImgs[start + imgind] for imgind in range(numImgs)]
        evalimgs = [img for img in evalimgs if img is not None]
        assert all(img['category_id'] == catId for img in evalimgs)
        # TODO assert all(img['maxDet'] is None for img in evalimgs)
        assert all(img['aRng'] == coco_eval.params.areaRng[0] for img in evalimgs)
        if not len(evalimgs):
            continue
        # again, we can ignore gtIgnore here, because we only look at areaRng[0]=all
        # again, we can ignore dtIgnore here, because we only look at maxDet=None
        numDTs = sum(len(img['dtIds']) for img in evalimgs)
        numGTs = sum(len(img['gtIds']) for img in evalimgs)
        overDTs = sum(len(img['dtIdsOver']) for img in evalimgs)
        overGTs = sum(len(img['gtIdsOver']) for img in evalimgs)
        underDTs = sum(len(img['dtIdsUnder']) for img in evalimgs)
        underGTs = sum(len(img['gtIdsUnder']) for img in evalimgs)
        numIoUs = sum(len(img['matches']) for img in evalimgs)
        numFPs = sum(len(img['dtMisses']) for img in evalimgs) + early_fp_counter
        numFNs = sum(len(img['gtMisses']) for img in evalimgs)
        sumIoUs = sum(sum(map(get(4), img['matches'])) for img in evalimgs) # sum(iou)
        sumIoGTs = sum(sum(map(get(2), img['matches'])) for img in evalimgs) # sum(iogt)
        sumIoDTs = sum(sum(map(get(3), img['matches'])) for img in evalimgs) # sum(iodt)
        sumTParea = sum(sum(map(get(5), img['matches'])) for img in evalimgs) # sum(inter)
        sumFParea = sum(sum(map(get(1), img['dtMisses'])) for img in evalimgs) # sum(area)
        sumFNarea = sum(sum(map(get(1), img['gtMisses'])) for img in evalimgs) # sum(area)
        if numDTs and numGTs:
            oversegmentation = overDTs * overGTs / numDTs / numGTs
            undersegmentation = underDTs * underGTs / numDTs / numGTs
            precision = (numDTs - numFPs) / numDTs
            recall = (numGTs - numFNs) / numGTs
        else:
            oversegmentation = undersegmentation = precision = recall = -1
        if numIoUs:
            iou = sumIoUs / numIoUs
            iogt = sumIoGTs / numIoUs
            iodt = sumIoDTs / numIoUs
        else:
            iou = iogt = iodt = -1
        if sumTParea or (sumFParea and sumFNarea):
            pixel_precision = sumTParea / (sumTParea + sumFParea)
            pixel_recall = sumTParea / (sumTParea + sumFNarea)
            pixel_iou = sumTParea / (sumTParea + sumFParea + sumFNarea)
        else:
            pixel_precision = pixel_recall = pixel_iou = -1
        catstats[cat['name']]['oversegmentation'] = oversegmentation
        catstats[cat['name']]['undersegmentation'] = undersegmentation
        catstats[cat['name']]['precision'] = precision
        catstats[cat['name']]['recall'] = recall
        catstats[cat['name']]['IoGT'] = iogt # i.e. per-match pixel-recall
        catstats[cat['name']]['IoDT'] = iodt # i.e. per-match pixel-precision
        catstats[cat['name']]['IoU'] = iou # i.e. per-match pixel-jaccardindex
        catstats[cat['name']]['pixel-precision'] = pixel_precision
        catstats[cat['name']]['pixel-recall'] = pixel_recall
        catstats[cat['name']]['pixel-iou'] = pixel_iou

    print("\nNEW RESULTS:\n")
    print(catstats)
    print("\n")
    coco_eval.accumulate()
    print("\nOLD COCO RESULTS\n")
    coco_eval.summarize()
    statInds = np.ones(12, np.bool)
    statInds[7] = False # AR maxDet[1]
    statInds[8] = False # AR maxDet[2]
    coco_eval.stats = coco_eval.stats[statInds]
    stats['scores'] = dict(zip([
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all ]',
        'Average Precision  (AP) @[ IoU=0.50      | area=   all ]',
        'Average Precision  (AP) @[ IoU=0.75      | area=   all ]',
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area= small ]',
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium ]',
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area= small ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area= large ]',
        ], coco_eval.stats.tolist()))
    return stats
