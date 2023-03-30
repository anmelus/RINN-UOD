import subprocess
import numpy as np

if __name__ == "__main__":
    # for i, gamma in enumerate(np.linspace(0, 0.1, 10)):
        # print(f"{i}/{10}: gamma = {gamma}")
        # result = subprocess.run(
        #     f"python main.py --model random --img val_images/* --add_bounding_boxes --num_clusters=5 --output_dir "
        #     f"eval_results/results_gamma/{gamma} --clustering kmeans --gamma={gamma} "
        #     f"--preds_filename=val_results/gamma/{gamma}.json",
        #     shell = True, stdout = subprocess.PIPE)
        # output = result.stdout.decode('utf-8')
        # f = open(f"eval_results/results_gamma/kmeans_{gamma}.txt", "w")
        # f.write(output)
        # f.close()

    # total = 16
    # for i, factor in enumerate(np.linspace(1, 2.5, total, endpoint = True)):
    #     print(f"{i}/{total}: factor = {factor}")
    #     result = subprocess.run(
    #         f"python main.py --model random --img val_images/* --add_bounding_boxes --num_clusters=5 --output_dir "
    #         f"eval_results/results_factor/{factor} --clustering agglomerative --factor={factor} "
    #         f"--preds_filename=val_results/factor/{factor}.json",
    #         shell = True, stdout = subprocess.PIPE)
    #     output = result.stdout.decode('utf-8')
    #     f = open(f"eval_results/results_factor/agglomerative_{factor}.txt", "w")
    #     f.write(output)
    #     f.close()


    # total = 11
    # for i, threshold in enumerate(np.linspace(0, 1, total, endpoint = True)):
    #     print(f"{i}/{total}: threshold = {threshold}")
    #     result = subprocess.run(
    #         f"python main.py --model random --img val_images/* --add_bounding_boxes --num_clusters=5 --output_dir "
    #         f"eval_results/results_threshold/{threshold} --clustering kmeans --threshold={threshold} "
    #         f"--preds_filename=val_results/threshold/{threshold}.json",
    #         shell = True, stdout = subprocess.PIPE)
    #     output = result.stdout.decode('utf-8')
    #     f = open(f"eval_results/results_threshold/kmeans_{threshold}.txt", "w")
    #     f.write(output)
    #     f.close()

    # Random
    # total = 9
    # for i, num_clusters in enumerate(np.arange(2, 11)):
    #     print(f"{i}/{total}: num_clusters = {num_clusters}")
    #     result = subprocess.run(
    #         f"python main.py --model random --img val_images/* --add_bounding_boxes --num_clusters={num_clusters} --output_dir "
    #         f"eval_results/results_num_clusters_random/{num_clusters} --clustering kmeans "
    #         f"--preds_filename=val_results/random_num_clusters/{num_clusters}.json",
    #         shell = True, stdout = subprocess.PIPE)
    #     output = result.stdout.decode('utf-8')
    #     f = open(f"eval_results/results_random_num_clusters/kmeans_{num_clusters}.txt", "w")
    #     f.write(output)
    #     f.close()

    # Trained
    # total = 9
    # for i, num_clusters in enumerate(np.arange(2, 11)):
    #     print(f"{i}/{total}: num_clusters = {num_clusters}")
    #     result = subprocess.run(
    #         f"python main.py --model trained --img val_images/* --add_bounding_boxes --num_clusters={num_clusters} --output_dir "
    #         f"eval_results/results_num_clusters_trained/{num_clusters} --clustering kmeans "
    #         f"--preds_filename=val_results/trained_num_clusters/kmeans_{num_clusters}.json",
    #         shell = True, stdout = subprocess.PIPE)
    #     output = result.stdout.decode('utf-8')
    #     f = open(f"eval_results/results_trained_num_clusters/kmeans_{num_clusters}.txt", "w")
    #     f.write(output)
    #     f.close()

    # Redo everything for image space?


    # Image space
    for i, gamma in enumerate(np.linspace(0.15, 1, 5)):
        print(f"{i}/{10}: gamma = {gamma}")
        result = subprocess.run(
            f"python main.py --model image_space --img val_images/* --add_bounding_boxes --num_clusters=3 --output_dir "
            f"eval_results/results_gamma_image_space/{gamma} --clustering kmeans --gamma={gamma} "
            f"--preds_filename=val_results/gamma_image_space/{gamma}.json",
            shell = True, stdout = subprocess.PIPE)
        output = result.stdout.decode('utf-8')
        f = open(f"eval_results/results_gamma_image_space/kmeans_{gamma}.txt", "w")
        f.write(output)
        f.close()


    # total = 5
    # for i, factor in enumerate(np.linspace(2.6, 3.5, total, endpoint = True)):
    #     print(f"{i}/{total}: factor = {factor}")
    #     result = subprocess.run(
    #         f"python main.py --model image_space --img val_images/* --add_bounding_boxes --num_clusters=3 --output_dir "
    #         f"eval_results/results_factor_image_space/{factor} --clustering agglomerative --factor={factor} "
    #         f"--preds_filename=val_results/factor_image_space/{factor}.json",
    #         shell = True, stdout = subprocess.PIPE)
    #     output = result.stdout.decode('utf-8')
    #     f = open(f"eval_results/results_factor_image_space/agglomerative_{factor}.txt", "w")
    #     f.write(output)
    #     f.close()

    # total = 9
    # for i, num_clusters in enumerate(np.arange(2, 11)):
    #     print(f"{i}/{total}: num_clusters = {num_clusters}")
    #     result = subprocess.run(
    #         f"python main.py --model image_space --img val_images/* --add_bounding_boxes --num_clusters={num_clusters} --output_dir "
    #         f"eval_results/results_num_clusters_image_space/{num_clusters} --clustering kmeans "
    #         f"--preds_filename=val_results/image_space_num_clusters/{num_clusters}.json",
    #         shell = True, stdout = subprocess.PIPE)
    #     output = result.stdout.decode('utf-8')
    #     f = open(f"eval_results/results_num_clusters_image_space/kmeans_{num_clusters}.txt", "w")
    #     f.write(output)
    #     f.close()
