import glob
import pickle
from collections import namedtuple, deque

import cv2
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
fontScale = 2
fontColor = (255, 0, 255)
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


def _create_samples(image, search_params, num_features):
    ystart = min([param.ystart for param in search_params])
    ystop = max([param.ystop for param in search_params])
    cropped = image[ystart:ystop, :, :]
    hog, hlsed = preprocess(cropped, vectorize=False, training=False)
    all_boxes = []
    samples = []
    # for scale, cells_per_step in scale2cells_per_step.items():
    for params in search_params:
        # samples_for_scale, boxes = _get_input_for_scale(scale, cropped, cells_per_step, hog, hlsed, ystart)
        samples_for_scale, boxes = _get_input_for_scale(cropped, hog, hlsed, params, num_features)
        # samples_for_scale, boxes = _get_randomized_input(cropped, hog, hlsed, params, num_features)
        samples.extend(samples_for_scale)
        all_boxes.extend(boxes)

    return samples, all_boxes


def _get_input_for_scale(image, hog, hlsed, params, num_features):
    window = int(original_size * params.scale)
    pixels_per_cell = int(training_pixels_per_cell * params.scale)
    num_horizontal_blocks = (image.shape[1] // pixels_per_cell) - cells_per_block + 1
    num_vertical_blocks = ((params.ystop - params.ystart) // pixels_per_cell) - cells_per_block + 1
    num_blocks_per_window = (window // pixels_per_cell) - cells_per_block + 1
    num_horizontal_steps = (num_horizontal_blocks - num_blocks_per_window) // params.horizontal_step
    num_vertical_steps = (num_vertical_blocks - num_blocks_per_window) // params.vertical_step

    xpositions = []
    ypositions = []
    samples = []
    for horizontal_idx in range(num_horizontal_steps):
        for vertical_idx in range(num_vertical_steps):
            # if numpy.random.uniform()> 0.4:
            ypos = vertical_idx * params.vertical_step
            xpos = horizontal_idx * params.horizontal_step
            hog_features = hog[ypos:ypos + num_blocks_per_window, xpos:xpos + num_blocks_per_window].ravel()
            xleft = xpos * pixels_per_cell
            ytop = ypos * pixels_per_cell
            hls = resize(hlsed[ytop:ytop + window, xleft:xleft + window], (original_size, original_size))
            features = concatenate((hog_features,
                                    color_hist(hls, nbins=color_histogram_bins),
                                    bin_spatial(hls, spatial_bin_shape)))
            if len(features) == num_features:
                samples.append(features)
                xpositions.append(xpos)
                ypositions.append(ypos)

    xleft = (array(xpositions) * pixels_per_cell).astype(numpy.int32)
    ytop = (array(ypositions) * pixels_per_cell).astype(numpy.int32)
    boxes = [((xleft, yleft), (xright, yright))
             for xleft, yleft, xright, yright
             in zip(xleft, ytop + params.ystart, xleft + window, ytop + window + params.ystart)]

    return samples, boxes


def find_cars(image, search_param):
    img_tosearch = image[search_param.ystart:search_param.ystop, :, :]
    ctrans_tosearch = img_tosearch
    if search_param.scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = resize(ctrans_tosearch,
                                 (numpy.int(imshape[1] / search_param.scale),
                                  numpy.int(imshape[0] / search_param.scale)))

    window = 64
    nxblocks = (ctrans_tosearch.shape[1] // training_pixels_per_cell) - cells_per_block + 1
    nyblocks = (ctrans_tosearch.shape[0] // training_pixels_per_cell) - cells_per_block + 1
    nblocks_per_window = (window // training_pixels_per_cell) - cells_per_block + 1
    nxsteps = (nxblocks - nblocks_per_window) // search_param.horizontal_step
    nysteps = (nyblocks - nblocks_per_window) // search_param.vertical_step

    # hog, hlsed = preprocess(ctrans_tosearch, False, False)
    hog = preprocess(ctrans_tosearch, False, False)
    xpositions = []
    ypositions = []
    samples = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            # if numpy.random.uniform() > 0.5:
            ypos = yb * search_param.vertical_step
            xpos = xb * search_param.horizontal_step
            hog_features = hog[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            # xleft = xpos * training_pixels_per_cell
            # ytop = ypos * training_pixels_per_cell
            # subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))
            # spatial_features = bin_spatial(subimg, size=spatial_bin_shape)
            # hist_features = color_hist(subimg, nbins=color_histogram_bins)
            # features = concatenate((hog_features, hist_features, spatial_features))
            features = hog_features
            samples.append(features)
            xpositions.append(xpos)
            ypositions.append(ypos)

    scaled_window = int(window * search_param.scale)
    xleft = (array(xpositions) * training_pixels_per_cell * search_param.scale).astype(numpy.int32)
    ytop = (array(ypositions) * training_pixels_per_cell * search_param.scale).astype(numpy.int32)

    boxes = [((xleft, yleft), (xright, yright))
             for xleft, yleft, xright, yright
             in zip(xleft,
                    ytop + search_param.ystart,
                    xleft + scaled_window,
                    ytop + scaled_window + search_param.ystart)]

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
        # cv2.putText(image, str(car_number), bbox[0], font, fontScale, fontColor, lineType)

    return image


search_parameter = namedtuple('parameter', ['scale', 'horizontal_step', 'vertical_step', 'ystart', 'ystop'])

search_params = [
    # search_parameter(0.8, 4, 2, 400, 592),
    # search_parameter(1, 2, 2, 400, 480),  # not bad
    search_parameter(1.5, 2, 2, 400, 656),  # not bad
    search_parameter(2, 2, 2, 400, 656),
]


class Pipeline:
    def __init__(self, vehicle_classifier, debug=None):
        self._camera_matrix, self._distortion_coefs = get_calibration_results()
        self._vehicle_model = vehicle_classifier
        self._debug = debug
        self._windows = numpy.zeros(shape=438)
        # self._windows = numpy.zeros(shape=294)

    def __call__(self, image, **kwargs):
        undistorted = undistort(image, self._camera_matrix, self._distortion_coefs, None, None)
        all_samples = []
        all_boxes = []
        for search_param in search_params:
            samples, boxes = find_cars(undistorted, search_param)
            all_samples.extend(samples)
            all_boxes.extend(boxes)

        # for box in all_boxes:
        #     undistorted = cv2.rectangle(undistorted, tuple(box[0]), tuple(box[1]), (0, 0, 255), 6)

        if self._debug == ALL_BOXES:
            for box in all_boxes:
                undistorted = cv2.rectangle(undistorted, tuple(box[0]), tuple(box[1]), (0, 0, 255), 6)
            return undistorted

        # predictions = self._vehicle_model.predict(all_samples)
        # indices = predictions == 1

        scores = self._vehicle_model.decision_function(all_samples)
        # print('scores', scores.min(), scores.max())
        # if any(scores > 0):
        self._windows = 0.7 * self._windows + 0.3 * scores
        # print('windows', self._windows.min(), self._windows.max())
        indices = self._windows > 0
        all_boxes = array(all_boxes)
        boxes = all_boxes[indices]

        # for box in boxes:
        #     undistorted = cv2.rectangle(undistorted, tuple(box[0]), tuple(box[1]), (0, 255, 0), 6)

        if self._debug == BOXES_WITH_CARS:
            for box in boxes:
                undistorted = cv2.rectangle(undistorted, tuple(box[0]), tuple(box[1]), (255, 0, 0), 6)
            return undistorted

        heat = numpy.zeros(image.shape[:2])
        heat = add_heat(heat, boxes, scores[indices])
        # heat = add_heat(heat, boxes)
        heat = apply_threshold(heat, 0.5)
        labels = label(heat)
        main_track = _draw_labeled_bboxes(undistorted, labels)
        return main_track


def draw_rectangle(image, points):
    for xmid, ymid in points:
        image = cv2.rectangle(image, (xmid - 5, ymid - 5), (xmid + 5, ymid + 5), (255, 0, 0), 6)
    return image


def _distance(p1, p2):
    return numpy.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


if __name__ == '__main__':
    process_and_save_video('test_video.mp4', 'test_output-new.mp4',
                           Pipeline(vehicle_classifier=pickle.load(open('pipeline.pkl', 'rb'))))
    #
    process_and_save_video('project_video.mp4', 'project_output-new.mp4',
                           Pipeline(vehicle_classifier=pickle.load(open('pipeline.pkl', 'rb'))))



    # image = cv2.cvtColor(cv2.imread('test_images/test3.jpg'), cv2.COLOR_BGR2RGB)
    # # image = Pipeline(vehicle_classifier=pickle.load(open('pipeline.pkl', 'rb')), debug=BOXES_WITH_CARS)(image)
    # image = Pipeline(vehicle_classifier=pickle.load(open('pipeline.pkl', 'rb')), debug=BOXES_WITH_CARS)(image)
    # import matplotlib.pyplot as plt
    #
    # print(image.shape)
    # plt.imshow(image)
    # plt.show()
    # #
