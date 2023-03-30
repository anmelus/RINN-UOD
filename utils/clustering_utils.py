import numpy as np
import scipy
import selectivesearch
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering, KMeans


def sel_search(img):
    """
    Perform selective search based on https://github.com/AlpacaTechJP/selectivesearch/blob/develop/example/example.py
    """

    img_lbl, regions = selectivesearch.selective_search(
        np.array(img), scale = 500, sigma = 0.9, min_size = 10)

    candidates = set()
    for r in regions:
        # Excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # Excluding regions smaller than 2000 pixels
        if r['size'] < 2000:
            continue
        # Distorted rects
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:
            continue
        candidates.add(r['rect'])

    bboxes = [list(bbox) for bbox in candidates]

    return bboxes


def make_connectivity_matrix(n_x, n_y):
    """
    Computes the pixel connectivity matrix for an image of size n_x * n_y. This is the matrix A of shape (n_x *
    n_y) x (n_x * n_y) where A[i, j] = 1 if pixels i and j are neighbours (either up/down or diagonally), 0 otherwise
    """

    num_pixels = n_x * n_y
    vertices = np.arange(num_pixels).reshape((n_x, n_y))
    edges_self = np.vstack((vertices.ravel(), vertices.ravel()))
    edges_right = np.vstack((vertices[:, :-1].ravel(), vertices[:, 1:].ravel()))
    edges_down = np.vstack((vertices[:-1].ravel(), vertices[1:].ravel()))
    edges_diag_l_r = np.vstack((vertices[:-1, :-1].ravel(), vertices[1:, 1:].ravel()))
    edges_diag_r_l = np.vstack((vertices[1:, :-1].ravel(), vertices[:-1, 1:].ravel()))
    edges = np.hstack((edges_self, edges_right, edges_down, edges_diag_l_r, edges_diag_r_l))

    # Make edges symmetric
    x_coords, y_coords = edges
    edges = np.hstack((edges, np.vstack((y_coords, x_coords))))
    data = np.ones(edges.shape[1])
    x_coords, y_coords = edges

    sparse_affinity_matrix = scipy.sparse.coo_matrix((data, (x_coords, y_coords)), shape = (num_pixels, num_pixels))
    return sparse_affinity_matrix


def agglomerative_clustering(features, factor = 1.5):
    """
    Performs agglomerative/hierarchical clustering on the features
    """

    c, h, w = features.shape
    nb_pixels = h * w
    features = np.moveaxis(features, 0, -1)

    flattened_features = np.reshape(features, (-1, c))
    connectivity = make_connectivity_matrix(h, w)

    affinity = cdist(flattened_features, flattened_features, "euclidean")
    affinity = np.reshape(affinity, (nb_pixels, nb_pixels))

    # TODO put back?
    # We set the threshold to be a function of the standard deviation of the affinity (distance) matrix instead of a
    # constant value since the distances between points in the feature space (and the variance of these distances)
    # can vary a lot from image to image
    clustering = AgglomerativeClustering(
        n_clusters = None, affinity = "precomputed", linkage = "complete", connectivity = connectivity,
        distance_threshold = factor * np.std(affinity)  # TODO put back to value
    )

    # TODO remove/keep?
    # clustering = AgglomerativeClustering(
    #     n_clusters = 7, affinity = "precomputed", linkage = "complete", connectivity = connectivity
    # )

    clustering.fit(affinity)
    labels = np.reshape(clustering.labels_, (h, w)).squeeze()
    return labels


def add_locality_to_features(features, h, w, gamma = 0.03):
    """
    Adds locality information to features by adding the points' coordinates as separate features
    """

    # x and y coordinates of all points
    yi = np.repeat(np.arange(h).reshape((h, 1)), w, axis = 1)
    xi = np.repeat(np.arange(w).reshape((1, w)), h, axis = 0)

    # Hyperparameter to vary the importance of locality in the clustering
    xi = xi * gamma
    yi = yi * gamma

    return np.concatenate((features, xi.reshape((-1, 1)), yi.reshape((-1, 1))), axis = 1)


def kmeans_clustering(features, num_clusters, add_locality = False, gamma = 0.03):
    """
    Performs K-means clustering on the features
    """

    # features: c x h x w tensor
    c, h, w = features.shape

    features = np.moveaxis(np.reshape(features, (c, h * w)), 0, 1)

    if add_locality:
        features = add_locality_to_features(features, h, w, gamma = gamma)

    kmeans = KMeans(n_clusters = num_clusters).fit(features)
    labels = kmeans.labels_.reshape((h, w))

    return labels
