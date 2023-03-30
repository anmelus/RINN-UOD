from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

"""
This script can be run to verify that the COCO eval method correctly has 1.000 precision on the ground truths.

"""
whole_list = []

with open('./modified/modified_val_data_gt.json') as file:
    gt = json.load(file)

for x in gt['annotations']:
    data = {'image_id' : x['image_id'],
            'category_id' : 1,
            'bbox' : x['bbox'],
            'score' : 1.000
    }

    whole_list.append(data)

with open('./modified/gt_results.json', 'w') as file:
    json.dump(whole_list, file)

# Instantiate COCO ground truth dataset
cocoGt = COCO('./modified/modified_val_data_gt.json')
imgIds = cocoGt.getImgIds(catIds=1)
cocoDt = cocoGt.loadRes('./modified/gt_results.json')

# Specify a list of category names of interest
cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
cocoEval.params.catIds = [1]
cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
