import glob
import pickle
from collections import namedtuple
from typing import NamedTuple

import cv2
import numpy
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy.ndimage import label

from lesson_functions import color_hist, bin_spatial
from training import original_size, training_pixels_per_cell, cells_per_block, resize, \
    color_histogram_bins, spatial_bin_shape, preprocess

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
fontScale = 1
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


# def find_cars(image, ystart, ystop, pipeline, scale2cells_per_step):
def find_cars(image, pipeline, search_params):
    ystart = min([param.ystart for param in search_params])
    ystop = max([param.ystop for param in search_params])
    cropped = image[ystart:ystop, :, :]
    hog, hlsed = preprocess(cropped, vectorize=False, training=False)
    all_boxes = []
    samples = []
    # for scale, cells_per_step in scale2cells_per_step.items():
    for params in search_params:
        # samples_for_scale, boxes = _get_input_for_scale(scale, cropped, cells_per_step, hog, hlsed, ystart)
        samples_for_scale, boxes = _get_input_for_scale(cropped, hog, hlsed, params)
        samples.extend(samples_for_scale)
        all_boxes.extend(boxes)

    predictions = pipeline.predict(samples)
    indices = predictions == 1
    all_boxes = array(all_boxes)
    return all_boxes[indices]


# def _get_input_for_scale(scale, image, cells_per_step, hog, hlsed, ystart):
def _get_input_for_scale(image, hog, hlsed, params):
    window = int(original_size * params.scale)
    pixels_per_cell = int(training_pixels_per_cell * params.scale)
    num_horizontal_blocks = (image.shape[1] // pixels_per_cell) - cells_per_block + 1
    num_vertical_blocks = ((params.ystop - params.ystart) // pixels_per_cell) - cells_per_block + 1
    num_blocks_per_window = (window // pixels_per_cell) - cells_per_block + 1
    num_horizontal_steps = (num_horizontal_blocks - num_blocks_per_window) // params.cells_per_step
    num_vertical_steps = (num_vertical_blocks - num_blocks_per_window) // params.cells_per_step
    xpositions = []
    ypositions = []
    samples = []
    for horizontal_idx in range(num_horizontal_steps):
        for vertical_idx in range(num_vertical_steps):
            ypos = vertical_idx * params.cells_per_step
            xpos = horizontal_idx * params.cells_per_step
            hog_features = hog[ypos:ypos + num_blocks_per_window, xpos:xpos + num_blocks_per_window].ravel()
            xleft = xpos * pixels_per_cell
            ytop = ypos * pixels_per_cell
            hls = resize(hlsed[ytop:ytop + window, xleft:xleft + window], (original_size, original_size))
            features = concatenate((hog_features,
                                    color_hist(hls, nbins=color_histogram_bins),
                                    bin_spatial(hls, spatial_bin_shape)))
            samples.append(features)
            xpositions.append(xpos)
            ypositions.append(ypos)

    xleft = (array(xpositions) * pixels_per_cell).astype(numpy.int32)
    ytop = (array(ypositions) * pixels_per_cell).astype(numpy.int32)
    boxes = [((xleft, yleft), (xright, yright))
             for xleft, yleft, xright, yright
             in zip(xleft, ytop + params.ystart, xleft + window, ytop + window + params.ystart)]
    return samples, boxes


def add_heat(heatmap, bbox_list):
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap


def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap


def draw_labeled_bboxes(image, labels):
    for car_number in range(1, labels[1] + 1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = numpy.array(nonzero[0])
        nonzerox = numpy.array(nonzero[1])
        xmin = numpy.min(nonzerox)
        ymin = numpy.min(nonzeroy)
        xmax = numpy.max(nonzerox)
        ymax = numpy.max(nonzeroy)
        bbox = ((xmin, ymin), (xmax, ymax))
        cv2.rectangle(image, bbox[0], bbox[1], (0, 0, 255), 6)
    return image


class VehicleDetector:
    def __init__(self):
        pass


# scale2cells_per_step = {1: 2, 1.5: 2, 2: 2}
scale2cells_per_step = {1: 2, 1.5: 2}
search_parameter = namedtuple('parameter', ['scale', 'cells_per_step', 'ystart', 'ystop'])
search_params = [search_parameter(1, 2, 400, 540), search_parameter(1.5, 2, 440, 600)]


class Pipeline:
    def __init__(self, vehicle_classifier, line_model=None, debug=None):
        self._camera_matrix, self._distortion_coefs = get_calibration_results()
        self._segmentation_model = line_model
        self._vehicle_model = vehicle_classifier
        self._debug = debug

    def __call__(self, image, **kwargs):
        undistorted = undistort(image, self._camera_matrix, self._distortion_coefs, None, None)
        ystart = 400
        ystop = 656

        # boxes = find_cars(undistorted, ystart, ystop, self._vehicle_model, scale2cells_per_step)
        boxes = find_cars(undistorted, self._vehicle_model, search_params)
        heat = numpy.zeros(image.shape[:2])
        heat = add_heat(heat, boxes)
        heat = apply_threshold(heat, 0)
        labels = label(heat)
        main_track = draw_labeled_bboxes(undistorted, labels)
        return main_track


if __name__ == '__main__':
    process_and_save_video('test_video.mp4', 'test_output-new.mp4',
                           Pipeline(vehicle_classifier=pickle.load(open('pipeline.pkl', 'rb'))))

    process_and_save_video('project_video.mp4', 'project_output-new.mp4',
                           Pipeline(vehicle_classifier=pickle.load(open('pipeline.pkl', 'rb'))))
    #
