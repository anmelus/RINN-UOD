"""
Converts the COCO dataset ground truth json file such that all category_ids are the same.
Additionally scales bounding box coordinates as if it were projected onto a 256x256 version of the image.

Also added downloading of test annotations images

Modify the if(TOTAL_LENGTH) statement to change the number of images that will be downloaded.

Downloads images to the directory images/
Downloads test_images to the directory test_images/
"""

from pycocotools.coco import COCO
import requests
import json
import os
import random
import shutil
import numpy as np

seed = 4
random.seed(seed)
np.random.seed(seed)

try:
    shutil.rmtree("./images")
except:
    os.mkdir("./images")
else:
    os.mkdir("./images")
try:
    shutil.rmtree("./test_images")
except:
    os.mkdir("./test_images")
else:
    os.mkdir("./test_images")
    


################## DOWNLOADING VALIDATION SET IMAGES

def download_data(path, sample, new_path, img_path):
    with open(path) as file:
        data = json.load(file)

    for item in data['annotations']:
        item['category_id'] = 1

    """ Scale image ground truth bounding boxes to 256x256 for accurate IoU calculation """
    ims = []
    ans = []
    print("Modifying boundings boxes...")

    # Iterate over indices to sample images from
    for i in sample:

        item = data['images'][i]

        for x in data['annotations']:
            if int(item['id']) == x['image_id']:
                x['bbox'][0] = round(x['bbox'][0] * 256 / item['width'], 2)
                x['bbox'][1] = round(x['bbox'][1] * 256 / item['height'], 2)
                x['bbox'][2] = round(x['bbox'][2] * 256 / item['width'], 2)
                x['bbox'][3] = round(x['bbox'][3] * 256 / item['height'], 2)
                x['area'] = x['bbox'][2] * x['bbox'][3]
                ans.append(x)

        ims.append(item)

    # Convert all entries into a dict format to be dumped into the new JSON data.
    modified = {"info" : data['info'],
                "images" : ims,
                "annotations" : ans,
                "licenses" : data['licenses'],
                "categories" : data['categories']
                }

    with open(new_path, 'w') as file:
        json.dump(modified, file)

    coco = COCO(new_path)

    # Get the corresponding image ids and images using loadImgs
    imgIds = coco.getImgIds(catIds=1)
    images = coco.loadImgs(imgIds)

    # Save the images into a local folder
    print("Downloading Images...")
    for im in images:
        img_data = requests.get(im['coco_url']).content
        with open(f'./{img_path}/' + im['file_name'], 'wb') as handler:
            handler.write(img_data)

################### DOWNLOADING TEST SET IMAGES
def download_test(path):
    with open(path) as file:
        data = json.load(file)

    # Sampling random entries in data['images'], modify TOTAL_LENGTH to change how many images are downloaded
    TOTAL_LENGTH = 50

    ims = []
    # Iterate over indices to sample images from
    for i in random.sample(range(len(data['images'])-1), TOTAL_LENGTH):
        item = data['images'][i]
        ims.append(item)
        
    # Save the images into a local folder
    print("Downloading test images...")
    for im in ims:
        img_data = requests.get(im['coco_url']).content
        with open('./test_images/' + im['file_name'], 'wb') as handler:
            handler.write(img_data)

TOTAL_LENGTH = 550
with open('./modified/instances_val2017.json') as file:
    data = json.load(file)

sample = random.sample(range(len(data['images'])), TOTAL_LENGTH)
download_data('./modified/instances_val2017.json', sample[:50], "./modified/val.json", "val_images")
download_data('./modified/instances_val2017.json', sample[50:], "./modified/test.json", "test_images")
