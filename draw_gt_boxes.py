import cv2
from PIL import Image
import os
import json
import re
import time

def draw_gts(args, img_path, filename_id, output_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256,256))

    with open(args.gt_json, 'r') as file:
        data = json.load(file)
    anns = data['annotations']

    print(f"filename id: {filename_id}")
    # drawing bounding boxes on image
    for i in range(len(anns)):
        if anns[i]['image_id'] == int(filename_id):
            x1, y1, w, h = anns[i]['bbox']
            cv2.rectangle(img, (int(x1), int(y1)), (int(x1+w), int(y1+h)), (0, 0, 255), 2)

    cv2.imwrite(output_path, img)
