import glob

import cv2
import numpy
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.classification import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing.data import StandardScaler
from sklearn.svm import LinearSVC

from lesson_functions import color_hist, bin_spatial


def get_data():
    cars = glob.glob('data/vehicles/*/*.png')
    notcars = glob.glob('data/non-vehicles/*/*.png')

    paths = numpy.concatenate((notcars, cars))
    X = numpy.array(list(map(lambda x: cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB), paths)))
    y = numpy.array(([0] * len(notcars) + [1] * len(cars)))
    print(X.shape, y.shape)
    return X, y


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                                  visualise=True, feature_vector=False, block_norm='L2')
        return features, hog_image
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                       visualise=False, feature_vector=feature_vec, block_norm='L2')
        return features


# def preprocess(image, vectorize=True):
#     # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     # gray = contrast_normalization(gray)
#     # image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
#     # image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
#     # image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
#     orient = 9
#     pix_per_cell = 8
#     cell_per_block = 2
#     features = []
#     # for channel in range(image.shape[2]):
#     features.append(
#         get_hog_features(
#             image[:, :, 0],
#             orient,
#             pix_per_cell,
#             cell_per_block,
#             vis=False,
#             feature_vec=vectorize))
#     if vectorize:
#         return numpy.concatenate(features)
#     else:
#         return features

def preprocess(image, vectorize=True, training=True):
    # for_hog = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    # for_hog = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    for_hog = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2

    hog = get_hog_features(
        # for_hog[:, :, 0],
        for_hog,
        orient,
        pix_per_cell,
        cell_per_block,
        vis=False,
        feature_vec=vectorize)

    if training:
        hlsed = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        hist_features = color_hist(hlsed, nbins=32)
        spatial_features = bin_spatial(hlsed, (16, 16))
        return numpy.concatenate((hog, hist_features, spatial_features))
        # return numpy.concatenate((hog, hist_features))
    else:
        return hog


def train(X, y):
    random_state = numpy.random.randint(0, 100000, 1)[0]
    X = numpy.array(list(map(preprocess, X)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, train_size=0.8, random_state=random_state)
    print(X_train.shape)

    pipeline = make_pipeline(
        # PCA(n_components=500, whiten=True),
        StandardScaler(),
        # LogisticRegression(C=0.001, n_jobs=-1, class_weight={0: 10, 1: 1})
        LinearSVC(C=0.001, class_weight={0: 10, 1: 1})
        # LinearSVC()
    )

    pipeline.fit(X_train, y_train)
    print(classification_report(y_test, pipeline.predict(X_test)))
    pipeline.fit(X, y)
    return pipeline


def train2(X, y):
    X = numpy.array(list(map(preprocess, X)))
    pipeline = make_pipeline(
        # PCA(n_components=500, whiten=True),
        StandardScaler(),
        # LogisticRegression(C=0.001, n_jobs=-1, class_weight={0: 5, 1: 1})
        # LinearSVC(C=0.01, class_weight={0: 10, 1: 1})
        LinearSVC(C=0.01, class_weight={0: 10, 1: 1})
    )

    pipeline.fit(X, y)
    return pipeline


def slide_window(image,
                 x_start_stop,
                 y_start_stop,
                 xy_window=(64, 64),
                 xy_overlap=(0.5, 0.5)):
    if x_start_stop is None:
        x_start_stop = [None, None]
    if y_start_stop is None:
        y_start_stop = [None, None]

    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = image.shape[1]
    if y_start_stop[0] == None:
        # y_start_stop[0] = int(image.shape[0] / 3)  # search in lower half of image
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = image.shape[0]

        # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = numpy.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = numpy.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = numpy.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = numpy.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = numpy.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = numpy.int((yspan - ny_buffer) / ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


def search_windows(image, windows, pipeline):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_image = cv2.resize(image[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = preprocess(test_image)
        prediction = pipeline.predict([features])
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    return on_windows


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    imcopy = numpy.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy


if __name__ == '__main__':
    X, y = get_data()
    pipeline = train(X, y)

    with open('pipeline.pkl', 'wb') as f:
        import pickle

        pickle.dump(pipeline, f)

    image = X[12234]

    # image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # orient = 8
    # pix_per_cell = 8
    # cell_per_block = 2
    # print(image.max())
    # # features, hog_image = get_hog_features(image[:, :, 0], orient,
    # features, hog_image = get_hog_features(image, orient,
    #                                        pix_per_cell, cell_per_block,
    #                                        vis=True, feature_vec=False)
    #
    # # Plot the examples
    # fig = plt.figure()
    # plt.subplot(121)
    # plt.imshow(image, cmap='gray')
    # plt.title('Example Car Image')
    # plt.subplot(122)
    # plt.imshow(hog_image, cmap='gray')
    # plt.title('HOG Visualization')
    # plt.show()

    # image = cv2.cvtColor(cv2.imread('test_images/test1.jpg'), cv2.COLOR_BGR2RGB)
    # draw_image = numpy.copy(image)
    # y_start_stop = [None, None]
    #
    # windows = slide_window(image,
    #                        x_start_stop=[None, None],
    #                        y_start_stop=y_start_stop,
    #                        xy_window=(128, 128),
    #                        xy_overlap=(0.5, 0.5))
    #
    # hot_windows = search_windows(image, windows, pipeline)
    #
    # window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
    #
    # plt.imshow(window_img)
    # plt.show()
