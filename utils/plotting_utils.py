import json
import os

import cv2
import numpy as np
from PIL.Image import Image
from matplotlib import pyplot as plt

from utils.bbox_computation_utils import select_bboxes_heuristic
from utils.evaluation_utils import MAP


def plot_clusters(img, labels, write_file = None):
    """
    Plots the clusters by overlaying them onto the image
    """

    # img: numpy array h x w x c
    h, w = labels.shape

    # s^2 is the number of pixels in the image that correspond to one pixel in the features/labels
    s = img.size[0] // h

    # The feature map may be in a lower dimension than the original image, in that case, we scale the cluster
    # assignments back to the original image dimensions. This is the exact operation done by a Kronecker product with
    # a block array of 1's
    scaled_labels = np.kron(labels, np.ones((s, s)))

    # Save the original image with its clusters
    if write_file is not None:
        plt.imshow(scaled_labels, cmap = 'jet')
        plt.imshow(img, alpha = 0.5)
        plt.axis('off')
        plt.savefig(write_file, bbox_inches = 'tight')


def plot_coco_gt(img_path, json_file, write_file = None):
    """
    Plots CoCo ground truth images and corresponding bounding boxes
    """

    img = Image.open(img_path.convert('RGB'))
    filename = os.path.split(img_path)[1]
    with open(json_file, 'r') as f:
        data = json.load(f)
    anns = data['annotations']

    # Finding image id corresponding to input image
    id = None
    for i in range(len(data['images'])):
        if data['images'][i]['file_name'] == filename:
            id = data['images'][i]['id']
            break

    # Drawing bounding boxes on image
    for i in range(len(anns)):
        if anns[i]['image_id'] == id:
            x1, y1, w, h = anns[i]['bbox']
            cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0, 0, 255), 2)

    if write_file is not None:
        cv2.imwrite(write_file, img)


def plot_bboxes(img, bboxes, file_name_id, preds_filename, write_file = None, use_heuristic = True, threshold = 0.2, reached_end=False):
    """
    Plots the given bounding boxes onto the image
    """
    img = np.array(img)

    if use_heuristic:
        bboxes = select_bboxes_heuristic(img, bboxes, threshold = threshold)

    draw_bboxes(img, bboxes, write_file)


    print(file_name_id)
    MAP(bboxes, file_name_id, preds_filename, reached_end)


def draw_bboxes(img, bboxes, write_file, color = (255, 0, 0)):
    """
    Draws all the bounding boxes in the given color on the image
    """

    for bbox in bboxes:
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),
                      (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), color, 2)

    # Draw the bounding boxes on the image
    cv2.imwrite(write_file, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
