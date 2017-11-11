import pickle
from collections import namedtuple, deque

import cv2
import keras
import numpy
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy.ndimage import label

from training import training_pixels_per_cell, cells_per_block, resize, color_histogram_bins, preprocess, original_size, \
    color_hist

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


def _create_samples_for_svm(image, search_param):
    part_to_search = image[search_param.ystart:search_param.ystop, :, :]
    if search_param.scale != 1:
        imshape = part_to_search.shape
        part_to_search = resize(part_to_search,
                                (numpy.int(imshape[1] / search_param.scale),
                                 numpy.int(imshape[0] / search_param.scale)))

    num_horizontal_blocks = (part_to_search.shape[1] // training_pixels_per_cell) - cells_per_block + 1
    num_vertical_blocks = (part_to_search.shape[0] // training_pixels_per_cell) - cells_per_block + 1
    blocks_per_window = (original_size // training_pixels_per_cell) - cells_per_block + 1
    num_horizontal_steps = (num_horizontal_blocks - blocks_per_window) // search_param.horizontal_step
    num_vertical_steps = (num_vertical_blocks - blocks_per_window) // search_param.vertical_step

    hog1, hog2, hlsed = preprocess(part_to_search, False, False)

    samples = []
    for xb in range(num_horizontal_steps):
        for yb in range(num_vertical_steps):
            ypos = yb * search_param.vertical_step
            xpos = xb * search_param.horizontal_step

            hog_features = numpy.concatenate((
                hog1[ypos:ypos + blocks_per_window, xpos:xpos + blocks_per_window].ravel(),
                hog2[ypos:ypos + blocks_per_window, xpos:xpos + blocks_per_window].ravel()))

            xleft = xpos * training_pixels_per_cell
            ytop = ypos * training_pixels_per_cell

            subimg = cv2.resize(hlsed[ytop:ytop + original_size, xleft:xleft + original_size], (64, 64))
            hist_features = color_hist(subimg, nbins=color_histogram_bins)
            samples.append(concatenate((hog_features, hist_features)))

    return samples


def _create_samples_for_cnn(image, search_params):
    ystart = min([param.ystart for param in search_params])
    ystop = max([param.ystop for param in search_params])
    cropped = image[ystart:ystop, :, :]
    samples = []

    for params in search_params:
        samples.extend(_get_input_for_scale(cropped, params))

    return samples


def _get_input_for_scale(image, params):
    window = int(original_size * params.scale)
    horizontal_step = int(params.horizontal_step * window)
    vertical_step = int(params.vertical_step * window)
    num_horizontal_steps = (image.shape[1] - window) // horizontal_step + 1
    num_vertical_steps = (image.shape[0] - window) // vertical_step + 1
    samples = []
    for horizontal_idx in range(num_horizontal_steps):
        for vertical_idx in range(num_vertical_steps):
            y_start = vertical_idx * vertical_step
            x_start = horizontal_idx * horizontal_step
            y_end = y_start + window
            x_end = x_start + window

            samples.append(resize(image[y_start:y_end, x_start:x_end], (original_size, original_size)))
    return samples


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


def _find_label_boundaries(labels):
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


class VehicleDetector:
    def __init__(self, vehicle_classifier, search_params, debug=None, return_boundaries=False):
        self._vehicle_model = vehicle_classifier
        self._debug = debug
        self._search_params = search_params
        self._return_boundaries = return_boundaries

    def __call__(self, image):
        if not hasattr(self, '_box_boundaries'):
            self._box_boundaries = self._calculate_box_boundaries(image)

        if self._debug == ALL_BOXES:
            return _draw_boxes_on(self._box_boundaries, image, (0, 0, 255))

        samples = self._get_samples(image)

        boxes = self._select_boxes_with_car(samples, self._box_boundaries)

        if self._debug == BOXES_WITH_CARS:
            return _draw_boxes_on(boxes, image, (0, 255, 0))

        heat = self._get_heatmap(image.shape[:2], boxes)
        if self._debug == HEATMAP:
            zero = numpy.zeros(heat.shape)
            return numpy.clip(numpy.dstack((heat, zero, zero)) * 2, 0, 255)

        labels = label(heat)
        boundaries = _find_label_boundaries(labels)
        if self._return_boundaries:
            return boundaries
        else:
            return _draw_boxes_on(boundaries, image, (255, 0, 0))

    def _get_samples(self, image):
        pass

    def _select_boxes_with_car(self, samples, boxes):
        pass

    def _calculate_box_boundaries(self, image):
        pass

    def _get_heatmap(self, shape, boxes):
        return add_heat(numpy.zeros(shape), boxes)


class SVMVehicleDetector(VehicleDetector):
    def __init__(self, *args, num_frames=10, heatmap_threshold=0.6, **kwargs):
        super(SVMVehicleDetector, self).__init__(*args, **kwargs)
        self._heat = deque(maxlen=num_frames)
        self._heatmap_threshold = heatmap_threshold

    def _calculate_box_boundaries(self, image):
        all_boxes = []
        for search_param in self._search_params:
            part_to_search = image[search_param.ystart:search_param.ystop, :, :]
            if search_param.scale != 1:
                imshape = part_to_search.shape
                part_to_search = resize(part_to_search,
                                        (numpy.int(imshape[1] / search_param.scale),
                                         numpy.int(imshape[0] / search_param.scale)))

            nxblocks = (part_to_search.shape[1] // training_pixels_per_cell) - cells_per_block + 1
            nyblocks = (part_to_search.shape[0] // training_pixels_per_cell) - cells_per_block + 1
            nblocks_per_window = (original_size // training_pixels_per_cell) - cells_per_block + 1
            nxsteps = (nxblocks - nblocks_per_window) // search_param.horizontal_step
            nysteps = (nyblocks - nblocks_per_window) // search_param.vertical_step

            xpositions = []
            ypositions = []
            for xb in range(nxsteps):
                for yb in range(nysteps):
                    ypositions.append(yb * search_param.vertical_step)
                    xpositions.append(xb * search_param.horizontal_step)

            scaled_window = int(original_size * search_param.scale)
            xleft = (array(xpositions) * training_pixels_per_cell * search_param.scale).astype(numpy.int32)
            ytop = (array(ypositions) * training_pixels_per_cell * search_param.scale).astype(numpy.int32)

            all_boxes.extend([((xl, yl), (xr, yr))
                              for xl, yl, xr, yr in zip(xleft,
                                                        ytop + search_param.ystart,
                                                        xleft + scaled_window,
                                                        ytop + scaled_window + search_param.ystart)])

        return all_boxes

    def _get_samples(self, image):
        all_samples = []
        for search_param in self._search_params:
            all_samples.extend(_create_samples_for_svm(image, search_param))
        return all_samples

    def _select_boxes_with_car(self, samples, boxes):
        predictions = self._vehicle_model.predict(samples)
        indices = predictions == 1
        return array(boxes)[indices]

    def _get_heatmap(self, shape, boxes):
        heat = super(SVMVehicleDetector, self)._get_heatmap(shape, boxes)
        self._heat.append(heat)
        return apply_threshold(numpy.mean(self._heat, axis=0), self._heatmap_threshold)


class CNNVehicleDetector(VehicleDetector):
    def __init__(self, *args, weight_for_new_frame=0.3, **kwargs):
        super(CNNVehicleDetector, self).__init__(*args, **kwargs)
        self._weight_for_new_frame = weight_for_new_frame

    def _get_samples(self, image):
        return _create_samples_for_cnn(image, self._search_params)

    def _calculate_box_boundaries(self, image):
        ystart = min([param.ystart for param in self._search_params])
        ystop = max([param.ystop for param in self._search_params])
        cropped = image[ystart:ystop, :, :]
        all_boxes = []

        for params in self._search_params:
            window = int(original_size * params.scale)
            horizontal_step = int(params.horizontal_step * window)
            vertical_step = int(params.vertical_step * window)
            num_horizontal_steps = (cropped.shape[1] - window) // horizontal_step + 1
            num_vertical_steps = (cropped.shape[0] - window) // vertical_step + 1
            for horizontal_idx in range(num_horizontal_steps):
                for vertical_idx in range(num_vertical_steps):
                    y_start = vertical_idx * vertical_step
                    x_start = horizontal_idx * horizontal_step
                    y_end = y_start + window
                    x_end = x_start + window

                    all_boxes.append(((x_start, params.ystart + y_start), (x_end, params.ystart + y_end)))

        return all_boxes

    def _select_boxes_with_car(self, samples, boxes):
        predictions = self._vehicle_model.predict(array(samples)).squeeze()
        if not hasattr(self, '_probabilities_for_boxes'):
            self._probabilities_for_boxes = numpy.zeros(len(self._box_boundaries))

        self._probabilities_for_boxes = (1 - self._weight_for_new_frame) * self._probabilities_for_boxes \
                                        + self._weight_for_new_frame * predictions
        indices = self._probabilities_for_boxes > 0.5
        return array(boxes)[indices]


if __name__ == '__main__':
    process_and_save_video('test_video.mp4', 'test_output-hog.mp4',
                           SVMVehicleDetector(vehicle_classifier=pickle.load(open('pipeline.pkl', 'rb')),
                                              search_params=[
                                                  search_parameter(1.5, 2, 2, 400, 656),
                                                  search_parameter(2, 2, 2, 400, 656),
                                              ]))

    process_and_save_video('test_video.mp4', 'test_output-cnn.mp4',
                           CNNVehicleDetector(
                               vehicle_classifier=keras.models.load_model('car_models/model-28-0.995.h5'),
                               search_params=[
                                   search_parameter(1.5, 0.5, 0.5, 400, 464),
                                   search_parameter(2, 0.5, 0.5, 400, 592),
                               ],
                               weight_for_new_frame=0.3
                           ))

    process_and_save_video('project_video.mp4', 'project_output-hog.mp4',
                           SVMVehicleDetector(vehicle_classifier=pickle.load(open('pipeline.pkl', 'rb')),
                                              search_params=[
                                                  search_parameter(1.5, 2, 2, 400, 656),
                                                  search_parameter(2, 2, 2, 400, 656),
                                              ]))

    process_and_save_video('project_video.mp4', 'project_output-cnn.mp4',
                           CNNVehicleDetector(
                               vehicle_classifier=keras.models.load_model('car_models/model-28-0.995.h5'),
                               search_params=[
                                   search_parameter(1.5, 0.5, 0.5, 400, 464),
                                   search_parameter(2, 0.5, 0.5, 400, 592),
                               ],
                               weight_for_new_frame=0.3
                           ))
