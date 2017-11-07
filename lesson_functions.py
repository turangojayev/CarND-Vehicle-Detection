import cv2
import numpy as np

resize = cv2.resize
hstack = np.hstack


def bin_spatial(img, size=(32, 32)):
    color1 = resize(img[:, :, 0], size).ravel()
    color2 = resize(img[:, :, 1], size).ravel()
    color3 = resize(img[:, :, 2], size).ravel()
    return hstack((color1, color2, color3))


histogram = np.histogram
concatenate = np.concatenate


def color_hist(img, nbins=32):  # bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = histogram(img[:, :, 0], bins=nbins)
    channel2_hist = histogram(img[:, :, 1], bins=nbins)
    channel3_hist = histogram(img[:, :, 2], bins=nbins)

    # Concatenate the histograms into a single feature vector
    hist_features = concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features
