import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth


def embedding_post_process(embedding, bin_seg, delta_v):
    """
    First use mean shift to find dense cluster center.

    Arguments:
    ----------
    embedding: numpy [H, W, embed_dim]
    bin_seg: numpy [H, W], each pixel is 0 or 1, 0 for background pixel
    delta_v: coordinates within distance of 2*delta_v to cluster center are

    Return:
    ---------
    cluster_result: numpy [H, W], index of different lanes on each pixel
    """
    cluster_result = np.zeros(bin_seg.shape, dtype=np.int32)

    y_coords, x_coords = np.nonzero(bin_seg)
    embedding_reshaped = embedding[y_coords, x_coords]
    num_pixels = len(y_coords)
    if num_pixels==0:
        return cluster_result

    bandwidth = estimate_bandwidth(embedding_reshaped, quantile=0.2, n_samples=num_pixels//2)
    mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=False, n_jobs=-1)
    mean_shift.fit(embedding_reshaped)

    labels = mean_shift.labels_
    unique_labels = np.unique(labels)
    unique_labels = sorted(unique_labels, key=lambda x:-np.sum(labels==x)) # sort according to occurrence
    cluster_centers = mean_shift.cluster_centers_

    lane_idx = 1
    for i in range(len(unique_labels)):
        if lane_idx > 4:         # only search for 4 lanes
            break
        mask1 = np.linalg.norm(embedding_reshaped - cluster_centers[unique_labels[i]], ord=2, axis=1) < 2*delta_v
        mask2 = (labels == unique_labels[i])
        if not np.any(mask1 & mask2):
            continue
        y_i = y_coords[mask1 & mask2]
        x_i = x_coords[mask1 & mask2]
        cluster_result[y_i, x_i] = lane_idx
        lane_idx += 1

    return cluster_result



if __name__=="__main__":
    from sklearn.datasets.samples_generator import make_blobs
    centers = [[1, 1, 1, 4], [-1, -1, 0, 3], [1, -1, 1, 3]]
    X, _ = make_blobs(n_samples=20*20, centers=centers, cluster_std=0.6)
    X = X.reshape((20, 20, -1))
    bin_seg = np.random.rand(20,20)

    embedding_post_process(X, bin_seg, .3)