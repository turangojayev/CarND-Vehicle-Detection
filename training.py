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

resize = cv2.resize
histogram = numpy.histogram
concatenate = numpy.concatenate

training_pixels_per_cell = 8
original_size = 64
cells_per_block = 2
orient = 9
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


def color_hist(img, nbins=32):
    channel1_hist = histogram(img[:, :, 0], bins=nbins)
    channel2_hist = histogram(img[:, :, 1], bins=nbins)
    channel3_hist = histogram(img[:, :, 2], bins=nbins)

    return concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))


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
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

    hog1 = get_hog_features(
        ycrcb[:, :, 0],
        orient,
        training_pixels_per_cell,
        cells_per_block,
        vis=False,
        feature_vec=vectorize)

    hog2 = get_hog_features(
        ycrcb[:, :, 1],
        orient,
        training_pixels_per_cell,
        cells_per_block,
        vis=False,
        feature_vec=vectorize)

    if training:
        hist_features = color_hist(hlsed, nbins=color_histogram_bins)
        return numpy.concatenate((hog1, hog2, hist_features))
    else:
        return hog1, hog2, hlsed


def train(X, y):
    random_state = numpy.random.randint(0, 100000, 1)[0]
    X = numpy.array(list(map(preprocess, X)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, train_size=0.8, random_state=random_state)
    print(X_train.shape)

    pipeline = make_pipeline(
        StandardScaler(),
        LinearSVC(C=0.01, class_weight={0: 10, 1: 1})
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
