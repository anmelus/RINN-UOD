import subprocess
import numpy as np
import os

num_clusters_image_space = 3
gamma_image_space = 0.3625
factor_image_space = 2.5

num_clusters_trained = 3
num_clusters_random = 4
factor_agglomerative_non_image_space = 1.5
gamma_random = 0.03


if __name__ == "__main__":
    result = subprocess.run(
        f"python main.py --model selective_search --img ./test_images/* --add_bounding_boxes --output_dir "
        f"./test_results/results_selective_search "
        f"--preds_filename=./test_results/results_selective_search/selective_search.json",
        shell = True, stdout = subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    filename = f"./test_results/results_selective_search/selective_search.txt"
    os.makedirs(os.path.dirname(filename), exist_ok = True)
    f = open(filename, "w+")
    f.write(output)
    f.close()

    result = subprocess.run(
        f"python main.py --model image_space --img test_images/* --add_bounding_boxes --num_clusters={num_clusters_image_space} --output_dir "
        f"./test_results/results_image_space_kmeans --clustering=kmeans --gamma={gamma_image_space} "
        f"--preds_filename=test_results/results_image_space_kmeans/image_space_kmeans.json",
        shell = True, stdout = subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    f = open(f"./test_results/results_image_space_kmeans/image_space_kmeans.txt", "w+")
    f.write(output)
    f.close()


    result = subprocess.run(
        f"python main.py --model image_space --img test_images/* --add_bounding_boxes --output_dir "
        f"./test_results/results_image_space_agglomerative --clustering=agglomerative --factor={factor_image_space} "
        f"--preds_filename=test_results/results_image_space_agglomerative/image_space_agglomerative.json",
        shell = True, stdout = subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    f = open(f"./test_results/results_image_space_agglomerative/image_space_agglomerative.txt", "w+")
    f.write(output)
    f.close()


    result = subprocess.run(
        f"python main.py --model trained --img test_images/* --add_bounding_boxes --num_clusters={num_clusters_trained} --output_dir "
        f"./test_results/results_trained_kmeans --clustering=kmeans --gamma=0 "
        f"--preds_filename=test_results/results_trained_kmeans/trained_kmeans.json",
        shell = True, stdout = subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    f = open(f"./test_results/results_trained_kmeans/trained_kmeans.txt", "w+")
    f.write(output)
    f.close()


    result = subprocess.run(
        f"python main.py --model trained --img test_images/* --add_bounding_boxes --output_dir "
        f"./test_results/results_trained_agglomerative --clustering=agglomerative --factor={factor_agglomerative_non_image_space} "
        f"--preds_filename=test_results/results_trained_agglomerative/trained_agglomerative.json",
        shell = True, stdout = subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    f = open(f"./test_results/results_trained_agglomerative/trained_agglomerative.txt", "w+")
    f.write(output)
    f.close()


    result = subprocess.run(
        f"python main.py --model random --img test_images/* --add_bounding_boxes --num_clusters={num_clusters_random} --output_dir "
        f"./test_results/results_random_kmeans --clustering=kmeans --gamma={gamma_random} "
        f"--preds_filename=test_results/results_random_kmeans/random_kmeans.json",
        shell = True, stdout = subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    f = open(f"./test_results/results_random_kmeans/random_kmeans.txt", "w+")
    f.write(output)
    f.close()


    result = subprocess.run(
        f"python main.py --model random --img test_images/* --add_bounding_boxes --output_dir "
        f"./test_results/results_random_agglomerative --clustering=agglomerative --factor={factor_agglomerative_non_image_space} "
        f"--preds_filename=test_results/results_random_agglomerative/random_agglomerative.json",
        shell = True, stdout = subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    f = open(f"./test_results/results_random_agglomerative/random_agglomerative.txt", "w+")
    f.write(output)
    f.close()

    ##########################
    # Positional encoding (well actually no position)
    result = subprocess.run(
        f"python main.py --model image_space --img test_images/* --add_bounding_boxes --num_clusters={num_clusters_image_space} --output_dir "
        f"./test_results/results_image_space_kmeans_gamma_0 --clustering=kmeans --gamma=0 "
        f"--preds_filename=test_results/results_image_space_kmeans_gamma_0/image_space_kmeans_gamma_0.json",
        shell = True, stdout = subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    f = open(f"./test_results/results_image_space_kmeans_gamma_0/image_space_kmeans_gamma_0.txt", "w+")
    f.write(output)
    f.close()


    result = subprocess.run(
        f"python main.py --model random --img test_images/* --add_bounding_boxes --num_clusters={num_clusters_random} --output_dir "
        f"./test_results/results_random_kmeans_gamma_0 --clustering=kmeans --gamma=0 "
        f"--preds_filename=test_results/results_random_kmeans_gamma_0/random_kmeans_gamma_0.json",
        shell = True, stdout = subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    f = open(f"./test_results/results_random_kmeans_gamma_0/random_kmeans_gamma_0.txt", "w+")
    f.write(output)
    f.close()
