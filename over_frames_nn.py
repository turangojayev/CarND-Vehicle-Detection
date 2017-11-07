import glob
import pickle
from collections import namedtuple, deque
from typing import NamedTuple

import cv2
import keras
import numpy
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy.ndimage import label

from lesson_functions import color_hist, bin_spatial
from training import original_size, training_pixels_per_cell, cells_per_block, resize, \
    color_histogram_bins, spatial_bin_shape, preprocess, orient

ALL_BOXES = 'all_boxes'
BOXES_WITH_CARS = 'boxes_with_cars'
HEATMAP = 'heat'

COLUMNS = 9
ROWS = 6
rows = 720
columns = 1280
#
src = numpy.float32([[0, 700],
                     [515, 472],
                     [764, 472.],
                     [1280, 700.]])

dst = numpy.float32([[100, 710],
                     [100, 10],
                     [1180, 10],
                     [1180, 710]])

font = cv2.FONT_HERSHEY_SIMPLEX
position = (10, 100)
fontScale = 0.5
fontColor = (0, 0, 255)
lineType = 2

ym_per_pix = 3. / 72
xm_per_pix = 3.7 / 700

perspective_tr_matrix = cv2.getPerspectiveTransform(src, dst)
inverse_perspective_tr_matrix = cv2.getPerspectiveTransform(dst, src)
RGB = ['Red', 'Green', 'Blue']
array = numpy.array
concatenate = numpy.concatenate
undistort = cv2.undistort
polyfit = numpy.polyfit


def get_calibration_results(rows=ROWS, columns=COLUMNS):
    image_paths = glob.glob('camera_cal/*.jpg')
    per_image_object_points = numpy.zeros((columns * rows, 3), numpy.float32)
    per_image_object_points[:, :2] = numpy.mgrid[0:columns, 0:rows].T.reshape(-1, 2)

    object_points = []
    image_points = []
    for path in image_paths:
        gray = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        return_value, corners = cv2.findChessboardCorners(gray, (columns, rows), None)
        if return_value is True:
            object_points.append(per_image_object_points)
            image_points.append(corners)

    test_image = cv2.cvtColor(cv2.imread(image_paths[2]), cv2.COLOR_BGR2GRAY)

    return_value, camera_matrix, distortion_coefs, rotation_vectors, translation_vectors = \
        cv2.calibrateCamera(object_points, image_points, test_image.shape[:2], None, None)

    return camera_matrix, distortion_coefs


def process_and_save_video(input, output, pipeline):
    clip = VideoFileClip(input)
    white_clip = clip.fl_image(pipeline)
    white_clip.write_videofile(output, audio=False)


def convert_if_needed(image):
    if image.dtype == numpy.float32:
        image = numpy.uint8(image * 255)
    return image


convert_color = cv2.cvtColor


def _create_samples(image, search_params):
    ystart = min([param.ystart for param in search_params])
    ystop = max([param.ystop for param in search_params])
    cropped = image[ystart:ystop, :, :]
    all_boxes = []
    samples = []
    # for scale, cells_per_step in scale2cells_per_step.items():
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


def _draw_labeled_bboxes(image, labels):
    for car_number in range(1, labels[1] + 1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = numpy.array(nonzero[0])
        nonzerox = numpy.array(nonzero[1])
        xmin = numpy.min(nonzerox)
        ymin = numpy.min(nonzeroy)
        xmax = numpy.max(nonzerox)
        ymax = numpy.max(nonzeroy)
        bbox = ((xmin, ymin), (xmax, ymax))
        cv2.rectangle(image, bbox[0], bbox[1], (255, 0, 0), 6)
    return image


search_parameter = namedtuple('parameter', ['scale', 'horizontal_step', 'vertical_step', 'ystart', 'ystop'])

search_params = [
    search_parameter(1.5, 0.5, 0.5, 400, 464),
    search_parameter(2, 0.5, 0.5, 400, 592),
]


class Pipeline:
    def __init__(self, vehicle_classifier, debug=None):
        self._camera_matrix, self._distortion_coefs = get_calibration_results()
        # self._segmentation_model = line_model
        self._vehicle_model = vehicle_classifier
        self._debug = debug
        self._windows = numpy.zeros(shape=113)

    def __call__(self, image, **kwargs):
        # undistorted = undistort(image, self._camera_matrix, self._distortion_coefs, None, None)
        undistorted = image
        samples, all_boxes = _create_samples(undistorted, search_params)
        # samples, all_boxes = _create_samples(cv2.cvtColor(undistorted, cv2.COLOR_BGR2HLS), search_params)

        # for box in all_boxes:
        #     undistorted = cv2.rectangle(undistorted, tuple(box[0]), tuple(box[1]), (0, 0, 255), 6)

        if self._debug == ALL_BOXES:
            for box in all_boxes:
                undistorted = cv2.rectangle(undistorted, tuple(box[0]), tuple(box[1]), (0, 0, 255), 6)
            return undistorted

        predictions = self._vehicle_model.predict(array(samples)).squeeze()
        self._windows = 0.7 * self._windows + 0.3 * predictions
        # indices = predictions > 0.5
        indices = self._windows > 0.5

        all_boxes = array(all_boxes)
        boxes = all_boxes[indices]

        # for box in boxes:
        #     undistorted = cv2.rectangle(undistorted, tuple(box[0]), tuple(box[1]), (0, 255, 0), 6)

        if self._debug == BOXES_WITH_CARS:
            for box in boxes:
                undistorted = cv2.rectangle(undistorted, tuple(box[0]), tuple(box[1]), (255, 0, 0), 6)
            return undistorted

        heat = numpy.zeros(image.shape[:2])
        heat = add_heat(heat, boxes)

        # heat = apply_threshold(heat, 0)
        labels = label(heat)
        main_track = _draw_labeled_bboxes(undistorted, labels)
        return main_track


if __name__ == '__main__':
    # process_and_save_video('test_video.mp4', 'test_output-new.mp4',
    #                        Pipeline(vehicle_classifier=keras.models.load_model('nn.h5')))

    import sys

    process_and_save_video('project_video.mp4', 'project_output-new.mp4',
                           Pipeline(vehicle_classifier=keras.models.load_model(sys.argv[1])))
