import glob
import pickle

import cv2
import numpy
from skimage.feature import hog
from sklearn.metrics.classification import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing.data import StandardScaler
from sklearn.svm import LinearSVC

from lesson_functions import color_hist, bin_spatial

resize = cv2.resize
training_pixels_per_cell = 16
original_size = 64
cells_per_block = 2
orient = 12
color_histogram_bins = 32
spatial_bin_shape = (16, 16)


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
        return hog(img,
                   orientations=orient,
                   pixels_per_cell=(pix_per_cell, pix_per_cell),
                   cells_per_block=(cell_per_block, cell_per_block),
                   transform_sqrt=False,
                   visualise=True,
                   feature_vector=False,
                   block_norm='L2')

    else:
        return hog(img,
                   orientations=orient,
                   pixels_per_cell=(pix_per_cell, pix_per_cell),
                   cells_per_block=(cell_per_block, cell_per_block),
                   transform_sqrt=False,
                   visualise=False,
                   feature_vector=feature_vec,
                   block_norm='L2')


def preprocess(image, vectorize=True, training=True):
    hlsed = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    hog = get_hog_features(
        gray,
        orient,
        training_pixels_per_cell,
        cells_per_block,
        vis=False,
        feature_vec=vectorize)

    if training:
        hist_features = color_hist(hlsed, nbins=color_histogram_bins)
        spatial_features = bin_spatial(hlsed, spatial_bin_shape)
        return numpy.concatenate((hog, hist_features, spatial_features))
    else:
        return hog, hlsed


def train(X, y):
    random_state = numpy.random.randint(0, 100000, 1)[0]
    X = numpy.array(list(map(preprocess, X)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, train_size=0.8, random_state=random_state)
    print(X_train.shape)

    pipeline = make_pipeline(
        StandardScaler(),
        LinearSVC(C=0.01, class_weight={0: 3, 1: 1})
    )

    pipeline.fit(X_train, y_train)
    print(classification_report(y_test, pipeline.predict(X_test)))
    print(pipeline.score(X_test, y_test))
    pipeline.fit(X, y)
    return pipeline


if __name__ == '__main__':
    X, y = get_data()
    pipeline = train(X, y)

    with open('pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
