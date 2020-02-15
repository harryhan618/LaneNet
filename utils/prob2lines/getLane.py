import cv2
import numpy as np


def polyfit2coords_tusimple(lane_pred, crop_h=0, resize_shape=None, y_px_gap=20, pts=None, ord=2):
    if resize_shape is None:
        resize_shape = lane_pred.shape
        crop_h = 0
    h, w = lane_pred.shape
    H, W = resize_shape
    coordinates = []

    if pts is None:
        pts = round(H / 2 / y_px_gap)

    for i in [idx for idx in np.unique(lane_pred) if idx!=0]:
        ys_pred, xs_pred = np.where(lane_pred==i)

        poly_params = np.polyfit(ys_pred, xs_pred, deg=ord)
        ys = np.array([h-y_px_gap/(H-crop_h)*h*i for i in range(1, pts+1)])
        xs = np.polyval(poly_params, ys)

        y_min, y_max = np.min(ys_pred), np.max(ys_pred)
        coordinates.append([[int(x/w*W) if x>=0 and x<w and ys[i]>=y_min and ys[i]<=y_max else -1,
                             H-y_px_gap*(i+1)] for (x, i) in zip(xs, range(pts))])

    return coordinates
