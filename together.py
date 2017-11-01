import glob
import os
import pickle
from functools import partial

import cv2
import numpy
# from keras.layers import *
from matplotlib import pyplot as plt
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy.ndimage import label

from classifier_training import preprocess
from lesson_functions import color_hist, bin_spatial

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


def find_cars(image, ystart, ystop, scale, pipeline):
    # image = image.astype(numpy.float32)

    cropped = image[ystart:ystop, :, :]

    if scale != 1:
        imshape = cropped.shape
        cropped = cv2.resize(cropped, (numpy.int(imshape[1] / scale), numpy.int(imshape[0] / scale)))
        # plt.imshow(cropped)
        # plt.show()


    window = 64
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    pixels_per_cell = 8
    nxblocks = (cropped.shape[1] // pixels_per_cell) - cells_per_step + 1
    nyblocks = (cropped.shape[0] // pixels_per_cell) - cells_per_step + 1
    nblocks_per_window = (window // pixels_per_cell) - cells_per_step + 1
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    hog = preprocess(cropped, vectorize=False, training=False)

    samples = []
    xpositions = []
    ypositions = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step

            hog_features = hog[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()

            xleft = xpos * pixels_per_cell
            ytop = ypos * pixels_per_cell
            subimg = cv2.resize(cropped[ytop:ytop + window, xleft:xleft + window], (64, 64))
            hls = cv2.cvtColor(subimg, cv2.COLOR_RGB2HLS)
            inf_indices = numpy.isinf(hls)
            # print('inf_indices', inf_indices)
            hls[inf_indices] = 255
            hls[hls > 255] = 255
            hist_features = color_hist(hls, nbins=32)

            spatial_features = bin_spatial(hls, (16, 16))

            # samples.append(hog_features)
            samples.append(numpy.concatenate((hog_features, hist_features, spatial_features)))
            # samples.append(numpy.concatenate((hog_features, hist_features)))
            xpositions.append(xpos)
            ypositions.append(ypos)

    xpositions = array(xpositions)
    ypositions = array(ypositions)

    predictions = pipeline.predict(samples)
    indices = predictions == 1

    xleft = (xpositions[indices] * pixels_per_cell * scale).astype(numpy.int32)
    ytop = (ypositions[indices] * pixels_per_cell * scale).astype(numpy.int32)
    win_draw = numpy.int(window * scale)
    boxes = [((xl, yl), (xr, yr)) for xl, yl, xr, yr in
             zip(xleft, ytop + ystart, xleft + win_draw, ytop + win_draw + ystart)]

    return boxes


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
        bbox = ((numpy.min(nonzerox), numpy.min(nonzeroy)), (numpy.max(nonzerox), numpy.max(nonzeroy)))
        cv2.rectangle(image, bbox[0], bbox[1], (0, 0, 255), 6)
    return image


class VehicleDetector:
    def __init__(self):
        pass


class Pipeline:
    def __init__(self, vehicle_classifier, line_model=None, debug=None):
        self._camera_matrix, self._distortion_coefs = get_calibration_results()
        self._segmentation_model = line_model
        self._vehicle_model = vehicle_classifier
        self._debug = debug

    def __call__(self, image, **kwargs):
        undistorted = undistort(image, self._camera_matrix, self._distortion_coefs, None, None)

        ystart = 350
        ystop = 656

        # scale = 1
        # boxes = find_cars(undistorted, ystart, ystop, scale, self._vehicle_model)

        scale = 1.5
        # boxes.extend(find_cars(undistorted, ystart, ystop, scale, self._vehicle_model))
        boxes = find_cars(undistorted, ystart, ystop, scale, self._vehicle_model)

        scale = 2
        boxes.extend(find_cars(undistorted, ystart, ystop, scale, self._vehicle_model))
        # boxes = find_cars(undistorted, ystart, ystop, scale, self._vehicle_model)

        heat = numpy.zeros_like(image[:, :, 0]).astype(numpy.float)
        heat = add_heat(heat, boxes)
        heat = apply_threshold(heat, 2)
        heatmap = numpy.clip(heat, 0, 255)
        labels = label(heatmap)
        main_track = draw_labeled_bboxes(undistorted, labels)
        return main_track

        # _put_text(main_track, (left_fit[-1] + right_fit[-1]) / 2 - columns / 2,
        #           *self._lines.curve_coefficients_in_meters())
        #
        # additional = numpy.concatenate((warped, warped_binary_in_rgb, lines_drawn), axis=1)
        # additional = cv2.resize(additional, (1280, 240))
        # return numpy.concatenate((additional, main_track), axis=0)

    def _get_thresholded(self, image):
        result = self._segmentation_model.predict(image.reshape(1, *image.shape)).squeeze() * 255
        result = result.astype(numpy.uint8)
        result = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 0)
        return result // 255


if __name__ == '__main__':
    process_and_save_video('test_video.mp4', 'test_output.mp4',
                           Pipeline(vehicle_classifier=pickle.load(open('pipeline.pkl', 'rb'))))

    process_and_save_video('project_video.mp4', 'project_output.mp4',
                           Pipeline(vehicle_classifier=pickle.load(open('pipeline.pkl', 'rb'))))



    # process_and_save_video('test_video.mp4', 'test_output.mp4',
    #                        Pipeline(vehicle_classifier=pickle.load(open('pipeline.pkl', 'rb')),
    #                                 line_model=keras.models.load_model('model.h5',
    #                                                                    custom_objects={'Upsampling': Upsampling})))

    # process_and_save_video('project_video.mp4', 'project_output.mp4',
    #                        Pipeline(
    #                            vehicle_classifier=pickle.load(open('pipeline.pkl', 'rb')),
    #                            line_model=keras.models.load_model('model.h5',
    #                                                               custom_objects={'Upsampling': Upsampling})))
