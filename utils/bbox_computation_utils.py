import cv2
import numpy as np


# TODO rewrite this + cite reference + boxA is in camelCase...
def iou(box_1, box_2):
    """
    Computes the IoU (intersection over union) of box_1 and box_2
    """

    # Compute the coordinates of the intersection rectangle
    x1 = max(box_1[0], box_2[0])
    x2 = min(box_1[0] + box_1[2], box_2[0] + box_2[2])
    y1 = max(box_1[1], box_2[1])
    y2 = min(box_1[1] + box_1[3], box_2[1] + box_2[3])

    intersection = abs(max((x2 - x1, 0)) * max((y2 - y1), 0))

    box_1_area = box_1[2] * box_1[3]
    box_2_area = box_2[2] * box_2[3]

    union = box_1_area + box_2_area - intersection

    return intersection / union


def bbox_area(bbox):
    """
    Returns the pixel area of bbox
    """

    _, _, w, h = bbox
    return w * h


def make_bbox(contour):
    """
    Computes the smallest bounding box that surrounds the contour
    """

    contours_poly = cv2.approxPolyDP(contour, 3, True)
    return list(cv2.boundingRect(contours_poly))


def find_bboxes(img, nb_clusters, labels):
    """
    Finds the bounding boxes of the clusters assignments
    """

    h, w = labels.shape
    s = img.size[0] // h

    # The feature map may be in a lower dimension than the original image, in that case, we scale the cluster
    # assignments back to the original image dimensions. This is the exact operation done by a Kronecker product with
    # a block array of 1's
    scaled_labels = np.kron(labels, np.ones((s, s)))

    all_bboxes = []

    for k in range(nb_clusters):
        # Inspired from: https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
        # To use findContours, the cluster should be in white and the rest of the image should be in black
        cluster_images = np.where(scaled_labels == k, 255, 0)
        cluster_images = np.stack([cluster_images for _ in range(3)], axis = -1).astype(np.uint8)
        imgray = cv2.cvtColor(cluster_images, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 127, 255, 0)
        cluster_contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cluster_contours = [make_bbox(c) for c in cluster_contours]
        all_bboxes.extend(cluster_contours)

    return all_bboxes


def merge_bboxes(box_1, box_2):
    """
    Returns the bounding box that surrounds box_1 and box_2
    """

    # Compute the coordinates of the union rectangle
    x1 = min(box_1[0], box_2[0])
    x2 = max(box_1[0] + box_1[2], box_2[0] + box_2[2])
    y1 = min(box_1[1], box_2[1])
    y2 = max(box_1[1] + box_1[3], box_2[1] + box_2[3])
    w = x2 - x1
    h = y2 - y1
    return [x1, x2, w, h]


def select_bboxes_heuristic(img, bboxes, threshold = 0.1):
    """
    Selects which bounding boxes to keep according to a heuristic
    """

    # Sort the bounding boxes by their area to apply the heuristic more efficiently
    bboxes = sorted(bboxes, key = bbox_area, reverse = True)

    img_area = img.shape[0] * img.shape[1]

    # TODO remove merging stuff if unused
    finished_merge = False
    while not finished_merge:
        finished_merge = True
        # List of bounding boxes to exclude
        exclude_list = []
        add_list = []
        for i in range(len(bboxes)):
            bbox_i = bboxes[i]
            area = bbox_area(bbox_i)

            # Remove the bounding box if it's too small or too large
            # TODO should small also be expressed as percentage of img area?
            _, _, w, h = bbox_i
            # Removes bounding boxes that are very unlikely to be objects (most of the time de to noise in the process)
            if area > 0.9 * img_area or area < 250 or w <= 5 or h <= 5:
                exclude_list.append(i)
            else:
                for j in range(i + 1, len(bboxes)):
                    if j not in exclude_list:
                        bbox_j = bboxes[j]
                        if iou(bbox_i, bbox_j) > threshold:
                            # TODO put back or removev
                            # add_list.append(merge_bboxes(bbox_i, bbox_j))
                            # finished_merge = False
                            # Since the list is sorted, this will always keep only the largest bounding box (bbox_j has
                            # area <= bbox_i)
                            exclude_list.append(j)
                            # exclude_list.append(i)

        excluded = set(exclude_list)
        bboxes = [c for i, c in enumerate(bboxes) if i not in excluded]
        bboxes.extend(add_list)
    return bboxes
