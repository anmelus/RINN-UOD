import json
import os

from pycocotools.coco import COCO

all_predictions = []

# Instantiate COCO specifying the annotations json path
coco_gt = COCO('modified/test.json')
img_ids = coco_gt.getImgIds(catIds = 1)


def MAP(bounding_boxes, file_name_id, preds_filename, reached_end=False):
    """
    Receives bounding boxes of all images part of the val_data_gt.json file and converts them into a results json
    file for COCO AP evaluation.

    bounding_box : [x, y, width, height] Denotes where the bounding box will be drawn.

    file_name_id : Image ID

    """

    ids = set(img_ids)
    if file_name_id in ids:
        convBoxes(bounding_boxes, file_name_id)

    # Simple check for when every image has been read since the image ids are in increasing order
    if reached_end:
        with open(preds_filename, 'w+') as file:
            json.dump(all_predictions, file)
            print("Done!")


def convBoxes(bounding_boxes, image_id):
    """
    Given the predicted bounding boxes and ID of the image, obtain the COCO results json dict format and append to
    list for writing to JSON later.

    Data is appended to list first because appending to json forces an entire rewrite of file.

    """
    for i in range(len(bounding_boxes)):
        data = {'image_id':    image_id,
                'category_id': 1,
                'bbox':        bounding_boxes[i],
                'score':       1
                }

        all_predictions.append(data)
