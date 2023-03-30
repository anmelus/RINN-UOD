#!/bin/bash

filename="./datasets/coco2014/val2014/COCO_val2014_000000001063.jpg"
output_dir="./outputs"
json_file="./datasets/coco2014/annotations/instances_val2014.json"

echo 'Running trained model...'
python main.py --model trained --output_dir $output_dir --img $filename --plot_all_clusters

echo 'Running random model...'
python main.py --model random --output_dir $output_dir --img $filename --plot_all_clusters

echo 'Running image space clustering...'
python main.py --model image_space --output_dir $output_dir --img $filename --plot_all_clusters

echo 'Running selective search...'
python main.py --model selective_search --output_dir $output_dir --img $filename

echo 'Running gt...'
python main.py --run_gt --json_file $json_file --output_dir $output_dir --img $filename