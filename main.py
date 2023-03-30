import argparse
import re
import random

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pprint import pprint
from tqdm import tqdm

from utils.bbox_computation_utils import find_bboxes
from utils.clustering_utils import *
from utils.model_utils import create_model
from utils.plotting_utils import *
from utils.preprocessing_utils import *
from draw_gt_boxes import draw_gts

seed = 4
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)



def run_model(args, filename, filename_id, data, reached_end=False):
    if not os.path.exists(os.path.join(args.output_dir, filename_id)):
        os.makedirs(os.path.join(args.output_dir, filename_id), exist_ok = True)

    features = None
    img = load_img(filename)
    preproc_img = process_input(img, args.model)

    # Running the model
    if args.model == "trained" or args.model == "random":
        model = create_model(args.model)
        model.eval()
        features = model(preproc_img)
        features = features.squeeze(0).detach().numpy()
    elif args.model == "image_space":
        # Image space clustering in hsv case
        # Since the R, G, and B components of an object are correlated with the light hitting it,
        # they make object detection harder than if we use an HSV source:
        # H.D.Cheng, X.H.Jiang, Y.Sun, Jingli Wang, Color image segmentation: advances and prospects,
        # Pattern Recognition, Volume 34, Issue 12, 2001, Pages 2259 - 2281, ISSN 0031 - 3203,
        # https://doi.org/10.1016/S0031320300001497
        features = rgb2hsv(img)

    # Clustering
    if args.model != "selective_search":
        # Clustering using features from pretrained resnet18, random BYOC encoder or image space

        if args.clustering == "kmeans":
            # If the model is already trained, we should keep the features as they already are (since the model was
            # trained in a supervised fashion to generate these features) if the model isn't trained, adding locality
            # information is a useful heuristic for object detection using kmeans
            add_locality = args.model != "trained"

            num_clusters = args.num_clusters
            if args.gt_num_clusters:
                # Get the ground truth number of clusters
                num_clusters = 0

                # Count the number of clusters in the ground truth
                for x in data['annotations']:
                    if x['image_id'] == int(filename_id):
                        num_clusters += 1
                # TODO  remove
                if num_clusters == 0:
                    return
                num_clusters += 1
            labels = kmeans_clustering(features, num_clusters, add_locality = add_locality, gamma = args.gamma)
        else:
            # Agglomerative/hierarchical clustering
            if args.model == "image_space":
                # If running in image space, preprocess the image for it to have the format expected by the
                # agglomerative clustering method (and reduce its size for the operation to be computationally
                # feasible)
                features = preprocess_image_agglomerative(features)
            # Running agglomerative clustering
            labels = agglomerative_clustering(features, factor = args.factor)

        # Since even if we specify a number of clusters, one of the cluster centers in kmeans could be assigned to
        # no point
        num_clusters = len(np.unique(labels))

        # Plotting the clusters
        output_file = f"{filename_id}_{args.model}_{args.clustering}_clust{args.num_clusters}.png"
        plot_clusters(img, labels, os.path.join(args.output_dir, filename_id, output_file))

        # Compute bounding boxes
        bboxes = find_bboxes(img, num_clusters, labels)
    else:
        bboxes = sel_search(preproc_img)

    # Drawing the bounding boxes
    plot_bboxes(img, bboxes, int(filename_id), args.preds_filename,
                os.path.join(args.output_dir, filename_id, f"{filename_id}_with_contours.png"), threshold = args.threshold, reached_end = reached_end)

    # Plot gt bbox
    draw_gts(args, filename, filename_id, os.path.join(args.output_dir, filename_id, 'coco_gt.png'))


def evaluate(args):
    # Instantiate COCO ground truth dataset
    cocoGt = COCO(args.gt_json)
    imgIds = cocoGt.getImgIds(catIds = 1)
    cocoDt = cocoGt.loadRes(args.preds_filename)

    # Specify a list of category names of interest
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')

    # TODO should we change this?
    # cocoEval.params.iouThrs = np.linspace(0.5, 1, 10, endpoint = True)

    cocoEval.params.catIds = [1]
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


def eval_torchmetrics(gt_json, pred_json, thresholds = None):
    metric = MeanAveragePrecision(box_format = 'xywh', iou_thresholds = thresholds, max_detection_thresholds = [1, 10, 100])

    with open(gt_json, 'r') as f:
        gt = json.load(f)['annotations']
    with open(pred_json, 'r') as f:
        preds = json.load(f)

    image_ids = set()
    for i in range(len(gt)):
        image_ids.add(gt[i]['image_id'])

    for id in image_ids:
        gt_bbox = []
        for i in range(len(gt)):
            if gt[i]['image_id'] == id:
                gt_bbox.append(gt[i]['bbox'])

        preds_bbox = []
        for i in range(len(preds)):
            if preds[i]['image_id'] == id:
                preds_bbox.append(preds[i]['bbox'])

        num_preds = len(preds_bbox)
        pred_dict = {'boxes':  torch.tensor(preds_bbox), 'scores': torch.ones(num_preds),
                     'labels': torch.zeros(num_preds)}
        num_gt = len(gt_bbox)
        gt_dict = {'boxes': torch.tensor(gt_bbox), 'labels': torch.zeros(num_gt)}
        metric.update([pred_dict], [gt_dict])

    pprint(metric.compute())


def main():
    parser = argparse.ArgumentParser(description = 'Qualitative results')
    parser.add_argument('--model', default = None)
    parser.add_argument('--clustering', default = "kmeans")
    parser.add_argument('--num_clusters', default = 5, type = int)
    parser.add_argument('--output_dir', default = "")
    parser.add_argument('--img', help = "Path to input images", nargs = '+')
    parser.add_argument('--add_bounding_boxes', action = 'store_true', default = False)
    parser.add_argument('--run_gt', action = 'store_true', default = False)
    parser.add_argument('--gamma', default = 0.03, type = float)
    parser.add_argument('--factor', default = 1.5, type = float)
    parser.add_argument('--threshold', default = 0.1, type = float)
    parser.add_argument('--preds_filename', required = True)
    parser.add_argument('--gt_json', default="./modified/test.json")
    parser.add_argument('--gt_num_clusters', action = 'store_true', default = False)
    parser.add_argument('--eval_only', action = 'store_true', default = False)

    args = parser.parse_args()

    if not args.eval_only:
        with open('./modified/instances_val2017.json') as file:
            data = json.load(file)
        os.makedirs(args.output_dir, exist_ok = True)
        os.makedirs(os.path.dirname(args.preds_filename), exist_ok = True)
        file_progress = tqdm(total = len(args.img), leave = False)
        count = 0
        for filename in args.img:
            print(filename)
            file_progress.desc = f"Current file: {filename}"
            file_name_id = re.sub('\D', '', filename).lstrip("0")
            run_model(args, filename, file_name_id, data, reached_end=(count == len(args.img) - 1))
            file_progress.update(1)
            count += 1

    print("****thresholds [0.5, 0.95]****")
    eval_torchmetrics(args.gt_json, args.preds_filename)
    beg = 0.3
    end = 0.95
    thresholds = list(np.linspace(beg, end, int((end-beg)/0.05) + 1, endpoint=True))
    print("****thresholds [0.3, 0.95]****")
    eval_torchmetrics(args.gt_json, args.preds_filename, thresholds = thresholds)


if __name__ == "__main__":
    main()
