import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth


def embedding_post_process(embedding, bin_seg, band_width=1.5, max_num_lane=4):
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

    cluster_list = embedding[bin_seg>0]
    if len(cluster_list)==0:
        return cluster_result

    mean_shift = MeanShift(bandwidth=1.5, bin_seeding=True)
    mean_shift.fit(cluster_list)

    labels = mean_shift.labels_
    cluster_result[bin_seg>0] = labels + 1

    cluster_result[cluster_result > max_num_lane] = 0
    for idx in np.unique(cluster_result):
        if len(cluster_result[cluster_result==idx]) < 15:
            cluster_result[cluster_result==idx] = 0

    return cluster_result



if __name__=="__main__":
    from sklearn.datasets.samples_generator import make_blobs
    centers = [[1, 1, 1, 4], [-1, -1, 0, 3], [1, -1, 1, 3]]
    X, _ = make_blobs(n_samples=20*20, centers=centers, cluster_std=0.6)
    X = X.reshape((20, 20, -1))
    bin_seg = np.random.rand(20,20)

    embedding_post_process(X, bin_seg, .3)