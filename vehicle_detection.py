import pickle
from collections import namedtuple, deque

import cv2
import keras
import numpy
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy.ndimage import label

from lesson_functions import color_hist
from training import training_pixels_per_cell, cells_per_block, resize, color_histogram_bins, preprocess, original_size

convert_color = cv2.cvtColor
array = numpy.array
concatenate = numpy.concatenate
undistort = cv2.undistort
polyfit = numpy.polyfit

ALL_BOXES = 'all_boxes'
BOXES_WITH_CARS = 'boxes_with_cars'
HEATMAP = 'heat'

search_parameter = namedtuple('parameter', ['scale', 'horizontal_step', 'vertical_step', 'ystart', 'ystop'])


def process_and_save_video(input, output, pipeline):
    clip = VideoFileClip(input)
    white_clip = clip.fl_image(pipeline)
    white_clip.write_videofile(output, audio=False)


def _create_samples_and_boxes_for_svm(image, search_param):
    img_tosearch = image[search_param.ystart:search_param.ystop, :, :]
    ctrans_tosearch = img_tosearch
    if search_param.scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = resize(ctrans_tosearch,
                                 (numpy.int(imshape[1] / search_param.scale),
                                  numpy.int(imshape[0] / search_param.scale)))

    nxblocks = (ctrans_tosearch.shape[1] // training_pixels_per_cell) - cells_per_block + 1
    nyblocks = (ctrans_tosearch.shape[0] // training_pixels_per_cell) - cells_per_block + 1
    nblocks_per_window = (original_size // training_pixels_per_cell) - cells_per_block + 1
    nxsteps = (nxblocks - nblocks_per_window) // search_param.horizontal_step
    nysteps = (nyblocks - nblocks_per_window) // search_param.vertical_step

    hog1, hog2, hlsed = preprocess(ctrans_tosearch, False, False)
    xpositions = []
    ypositions = []
    samples = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * search_param.vertical_step
            xpos = xb * search_param.horizontal_step

            hog_features = numpy.concatenate((
                hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel(),
                hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()))

            xleft = xpos * training_pixels_per_cell
            ytop = ypos * training_pixels_per_cell

            subimg = cv2.resize(hlsed[ytop:ytop + original_size, xleft:xleft + original_size], (64, 64))

            hist_features = color_hist(subimg, nbins=color_histogram_bins)

            features = numpy.concatenate((hog_features, hist_features))
            samples.append(features)
            xpositions.append(xpos)
            ypositions.append(ypos)

    scaled_window = int(original_size * search_param.scale)
    xleft = (array(xpositions) * training_pixels_per_cell * search_param.scale).astype(numpy.int32)
    ytop = (array(ypositions) * training_pixels_per_cell * search_param.scale).astype(numpy.int32)

    boxes = [((xleft, yleft), (xright, yright))
             for xleft, yleft, xright, yright
             in zip(xleft,
                    ytop + search_param.ystart,
                    xleft + scaled_window,
                    ytop + scaled_window + search_param.ystart)]

    return samples, boxes


def _create_samples_and_boxes_for_cnn(image, search_params):
    ystart = min([param.ystart for param in search_params])
    ystop = max([param.ystop for param in search_params])
    cropped = image[ystart:ystop, :, :]
    all_boxes = []
    samples = []

    for params in search_params:
        samples_for_scale, boxes = _get_input_for_scale(cropped, params)
        samples.extend(samples_for_scale)
        all_boxes.extend(boxes)

    return samples, all_boxes


def _get_input_for_scale(image, params):
    window = int(original_size * params.scale)
    horizontal_step = int(params.horizontal_step * window)
    vertical_step = int(params.vertical_step * window)
    num_horizontal_steps = (image.shape[1] - window) // horizontal_step + 1
    num_vertical_steps = (image.shape[0] - window) // vertical_step + 1
    boxes = []
    samples = []
    for horizontal_idx in range(num_horizontal_steps):
        for vertical_idx in range(num_vertical_steps):
            y_start = vertical_idx * vertical_step
            x_start = horizontal_idx * horizontal_step
            y_end = y_start + window
            x_end = x_start + window

            samples.append(resize(image[y_start:y_end, x_start:x_end], (original_size, original_size)))
            boxes.append(((x_start, params.ystart + y_start), (x_end, params.ystart + y_end)))
    return samples, boxes


def add_heat(heatmap, boxes, scores=None):
    for i, box in enumerate(boxes):
        if scores is not None:
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += scores[i]
        else:
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap


def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap


def _draw_boxes_on(boxes, image, color=(0, 0, 255)):
    for box in boxes:
        image = cv2.rectangle(image, tuple(box[0]), tuple(box[1]), color, 6)
    return image


class VehicleDetector:
    def __init__(self, vehicle_classifier, search_params, debug=None):
        self._vehicle_model = vehicle_classifier
        self._debug = debug
        self._search_params = search_params

    def __call__(self, image):
        samples, boxes = self._get_samples_and_boxes(image)
        if self._debug == ALL_BOXES:
            return _draw_boxes_on(boxes, image, (0, 0, 255))

        boxes = self._select_boxes_with_car(samples, boxes)

        if self._debug == BOXES_WITH_CARS:
            return _draw_boxes_on(boxes, image, (0, 255, 0))

        heat = self._get_heatmap(image.shape[:2], boxes)
        if self._debug == HEATMAP:
            zero = numpy.zeros(heat.shape)
            return numpy.clip(numpy.dstack((heat, zero, zero)) * 10, 0, 255)

        labels = label(heat)
        boundaries = self._find_label_boundaries(labels)
        return _draw_boxes_on(boundaries, image, (255, 0, 0))

    def _get_samples_and_boxes(self, image):
        pass

    def _select_boxes_with_car(self, samples, boxes):
        pass

    def _get_heatmap(self, shape, boxes):
        return add_heat(numpy.zeros(shape), boxes)

    def _find_label_boundaries(self, labels):
        boundaries = []
        for car_number in range(1, labels[1] + 1):
            nonzero = (labels[0] == car_number).nonzero()
            nonzeroy = array(nonzero[0])
            nonzerox = array(nonzero[1])
            xmin = numpy.min(nonzerox)
            ymin = numpy.min(nonzeroy)
            xmax = numpy.max(nonzerox)
            ymax = numpy.max(nonzeroy)
            boundaries.append(((xmin, ymin), (xmax, ymax)))
        return boundaries


class SVMVehicleDetector(VehicleDetector):
    def __init__(self, vehicle_classifier, search_params, debug=None):
        super(SVMVehicleDetector, self).__init__(vehicle_classifier, search_params, debug)
        self._windows = numpy.zeros(shape=192)
        self._heat = deque(maxlen=10)

    def _get_samples_and_boxes(self, image):
        all_samples = []
        all_boxes = []
        for search_param in self._search_params:
            samples, boxes = _create_samples_and_boxes_for_svm(image, search_param)
            all_samples.extend(samples)
            all_boxes.extend(boxes)
        return all_samples, all_boxes

    def _select_boxes_with_car(self, samples, boxes):
        predictions = self._vehicle_model.predict(samples)
        indices = predictions == 1
        return array(boxes)[indices]

    def _get_heatmap(self, shape, boxes):
        heat = super(SVMVehicleDetector, self)._get_heatmap(shape, boxes)
        self._heat.append(heat)
        return apply_threshold(numpy.mean(self._heat, axis=0), 0.6)


class CNNVehicleDetector(VehicleDetector):
    def __init__(self, vehicle_classifier, search_params, debug=None):
        super(CNNVehicleDetector, self).__init__(vehicle_classifier, search_params, debug)
        self._windows = numpy.zeros(shape=113)

    def _get_samples_and_boxes(self, image):
        return _create_samples_and_boxes_for_cnn(image, self._search_params)

    def _select_boxes_with_car(self, samples, boxes):
        predictions = self._vehicle_model.predict(array(samples)).squeeze()
        self._windows = 0.7 * self._windows + 0.3 * predictions
        indices = self._windows > 0.5
        return array(boxes)[indices]


if __name__ == '__main__':
    # process_and_save_video('test_video.mp4', 'test_output-new.mp4',
    #                        Pipeline(vehicle_classifier=pickle.load(open('pipeline.pkl', 'rb'))))
    #
    # process_and_save_video('project_video.mp4', 'project_output-hog.mp4',
    #                        SVMVehicleDetector(vehicle_classifier=pickle.load(open('pipeline.pkl', 'rb')),
    #                                           search_params=[
    #                                               search_parameter(1.5, 2, 2, 400, 656),
    #                                               search_parameter(2, 2, 2, 400, 656),
    #                                           ]))


    process_and_save_video('project_video.mp4', 'project_output-cnn.mp4',
                           CNNVehicleDetector(
                               vehicle_classifier=keras.models.load_model('car_models/model-28-0.995.h5'),
                               search_params=[
                                   search_parameter(1.5, 0.5, 0.5, 400, 464),
                                   search_parameter(2, 0.5, 0.5, 400, 592),
                               ]
                           ))



    # image = cv2.cvtColor(cv2.imread('test_images/test3.jpg'), cv2.COLOR_BGR2RGB)
    # # image = Pipeline(vehicle_classifier=pickle.load(open('pipeline.pkl', 'rb')), debug=BOXES_WITH_CARS)(image)
    # image = Pipeline(vehicle_classifier=pickle.load(open('pipeline.pkl', 'rb')), debug=BOXES_WITH_CARS)(image)
    # import matplotlib.pyplot as plt
    #
    # print(image.shape)
    # plt.imshow(image)
    # plt.show()
    # #
