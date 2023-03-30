from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import rescale


def load_img(path):
    """Loads the given image, transforms it into RGB (in case it was in greyscale) and resizes it"""

    return Image.open(path).convert('RGB').resize((256, 256))


def rgb2hsv(img):
    """
    Converts the given image to HSV
    """

    # img: h x w x c PIL image
    return torch.tensor(np.transpose(img.convert('HSV'), axes = [2, 0, 1])).numpy()  # c x h x w


def process_input(img, model):
    """
    Computes the model-dependent image preprocessing
    """

    if model == "random":
        # For the random model, we only normalize the images
        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        img = preprocess(img)
        return (img - torch.mean(img, dim = (1, 2)).reshape((-1, 1, 1))).unsqueeze(0)
    elif model == "selective_search":
        # Selective search directly runs on the RGB image
        return np.array(img)
    else:
        # Running on the trained model, we normalize the images as they are expected in this format by the ResNet18
        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225]
            )])
        return preprocess(img).unsqueeze(0)


def preprocess_image_agglomerative(img):
    """
    Preprocesses the image to return it as expected by the agglomerative clustering algorithm, reducing its size and
    applying a Gaussian filter to reduce aliasing
    """
    _, h, w = img.shape

    # Normalize pixel values as expected by the clustering algorithm
    normalized_img = img / 255

    # From: https://scikit-learn.org/stable/auto_examples/cluster/plot_coin_ward_segmentation.html
    # The Gaussian filter is used to reduce aliasing artifacts and we reduce the image size to speed up clustering
    smoothed_features = gaussian_filter(normalized_img, sigma = 2)
    rescaled_features = rescale(
        smoothed_features,
        0.2,
        mode = "reflect",
        anti_aliasing = False,
    )
    return rescaled_features
