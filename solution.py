import glob
import os
from functools import partial

import cv2
import keras
import numpy
import tensorflow as tf
from keras.layers import *
from matplotlib import pyplot as plt
from moviepy.video.io.VideoFileClip import VideoFileClip

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


def perspective_transform(image, matrix):
    return cv2.warpPerspective(image, matrix, (columns, rows), flags=cv2.INTER_LINEAR)


def resize_images_bilinear(X, target_height=None, target_width=None):
    '''Resizes the images contained in a 4D tensor of shape
    - [batch, height, width, channels]
    '''
    new_shape = tf.constant(np.array((target_height, target_width)).astype('int32'))
    X = tf.image.resize_bilinear(X, new_shape)
    X.set_shape((None, target_height, target_width, None))
    return X


class Upsampling(Layer):
    def __init__(self, size=(1, 1), target_size=None, **kwargs):
        self.size = tuple(size)
        self.target_size = tuple(target_size)
        self.input_spec = [InputSpec(ndim=4)]
        super(Upsampling, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        width = self.target_size[0]
        height = self.target_size[1]
        return (input_shape[0], width, height, input_shape[3])

    def call(self, x, mask=None):
        return resize_images_bilinear(x, target_height=self.target_size[0], target_width=self.target_size[1])

    def get_config(self):
        config = {'size': self.size, 'target_size': self.target_size}
        base_config = super(Upsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Lines:
    def __init__(self, margin=60, minpix=30, num_windows=9, ym_per_pix=3. / 72, xm_per_pix=3.7 / 700, alpha=0.2):
        self._margin = margin
        self._minpix = minpix
        self._num_windows = num_windows
        self._image_shape = None
        self._ym_per_pix = ym_per_pix
        self._xm_per_pix = xm_per_pix
        self._alpha = alpha

    def get_lines(self, binary_image):
        nonzeroy, nonzerox = binary_image.nonzero()
        if self._image_shape is None:
            self._image_shape = binary_image.shape

        left_indices, right_indices = self._find_indices(binary_image, nonzerox, nonzeroy)
        y, left_fit, right_fit = self._fit_and_update(nonzerox[left_indices], nonzeroy[left_indices],
                                                      nonzerox[right_indices], nonzeroy[right_indices])

        return y, left_fit, right_fit, nonzeroy, nonzerox, left_indices, right_indices

    def curve_coefficients_in_meters(self):
        return self._left_coeffs_m, self._right_coeffs_m

    def _fit_and_update(self, leftx, lefty, rightx, righty):
        left_coeffs = polyfit(lefty, leftx, 2) if len(leftx) > 0 else self._left_coeffs
        right_coeffs = polyfit(righty, rightx, 2) if len(rightx) > 0 else self._right_coeffs

        left_coeffs_m = polyfit(lefty * self._ym_per_pix, leftx * self._xm_per_pix, 2) \
            if len(leftx) > 0 else self._left_coeffs_m

        right_coeffs_m = polyfit(righty * self._ym_per_pix, rightx * self._xm_per_pix, 2) \
            if len(rightx) > 0 else self._right_coeffs_m

        y = numpy.linspace(0, self._image_shape[0] - 1, self._image_shape[0])
        left_fit = left_coeffs[0] * y ** 2 + left_coeffs[1] * y + left_coeffs[2]
        right_fit = right_coeffs[0] * y ** 2 + right_coeffs[1] * y + right_coeffs[2]

        if not hasattr(self, '_left_coeffs'):
            self._left_coeffs = left_coeffs
            self._right_coeffs = right_coeffs
            self._left_coeffs_m = left_coeffs_m
            self._right_coeffs_m = right_coeffs_m
            self._start = 0

        else:
            closest_point_difference = right_fit[-1] - left_fit[-1]
            if closest_point_difference > 0:
                differences = right_fit - left_fit
                acceptable = differences > 0.7 * closest_point_difference
                start = numpy.argmax(acceptable)
                self._start += int(0.1 * (start - self._start))
                y = y[self._start:]

                if right_fit[2] - left_fit[2] > 350:
                    self._left_coeffs += self._alpha * (left_coeffs - self._left_coeffs)
                    self._right_coeffs += self._alpha * (right_coeffs - self._right_coeffs)
                    self._left_coeffs_m += self._alpha * (left_coeffs_m - self._left_coeffs_m)
                    self._right_coeffs_m += self._alpha * (right_coeffs_m - self._right_coeffs_m)

        left_fit = self._left_coeffs[0] * y ** 2 + self._left_coeffs[1] * y + self._left_coeffs[2]
        right_fit = self._right_coeffs[0] * y ** 2 + self._right_coeffs[1] * y + self._right_coeffs[2]
        return y, left_fit, right_fit

    def _find_indices(self, binary_image, nonzerox, nonzeroy):
        if not hasattr(self, '_left_coeffs'):
            histogram = numpy.sum(binary_image[self._image_shape[0] // 2:, :], axis=0)
            midpoint = numpy.int(histogram.shape[0] / 2)
            self._leftx_current = numpy.argmax(histogram[:midpoint])
            self._rightx_current = numpy.argmax(histogram[midpoint:]) + midpoint

            window_height = numpy.int(self._image_shape[0] / self._num_windows)

            left_indices, right_indices = [], []

            for window_idx in range(self._num_windows):
                left_within_window_indices, right_within_window_indices = self._process_window(nonzerox,
                                                                                               nonzeroy,
                                                                                               window_idx,
                                                                                               window_height)
                left_indices.append(left_within_window_indices)
                right_indices.append(right_within_window_indices)

            left_indices = numpy.concatenate(left_indices)
            right_indices = numpy.concatenate(right_indices)

        else:
            old_leftx = self._left_coeffs[0] * (nonzeroy ** 2) + self._left_coeffs[1] * nonzeroy + self._left_coeffs[2]
            left_indices = ((nonzerox > (old_leftx - self._margin)) & (nonzerox < (old_leftx + self._margin)))

            old_rightx = self._right_coeffs[0] * (nonzeroy ** 2) + self._right_coeffs[1] * nonzeroy + \
                         self._right_coeffs[2]

            right_indices = ((nonzerox > (old_rightx - self._margin)) & (nonzerox < (old_rightx + self._margin)))
        return left_indices, right_indices

    def _process_window(self, nonzerox, nonzeroy, window_idx, window_height):
        # lower and upper part of screen, origin is at top left of screen
        window_bottom = self._image_shape[0] - window_idx * window_height
        window_top = window_bottom - window_height
        left_window_left = self._leftx_current - self._margin
        left_window_right = self._leftx_current + self._margin
        right_window_left = self._rightx_current - self._margin
        right_window_right = self._rightx_current + self._margin

        within_vertical_boundaries = (nonzeroy >= window_top) & (nonzeroy < window_bottom)
        within_left_window_horizontal_boundaries = (nonzerox >= left_window_left) & (nonzerox < left_window_right)
        within_right_window_horizontal_boundaries = (nonzerox >= right_window_left) & (nonzerox < right_window_right)

        left_within_window_indices = (within_vertical_boundaries &
                                      within_left_window_horizontal_boundaries).nonzero()[0]

        right_within_window_indices = (within_vertical_boundaries &
                                       within_right_window_horizontal_boundaries).nonzero()[0]

        # If more than minpix pixels were found, recenter next window on their mean position
        if len(left_within_window_indices) > self._minpix:
            self._leftx_current = numpy.int(numpy.mean(nonzerox[left_within_window_indices]))
        if len(right_within_window_indices) > self._minpix:
            self._rightx_current = numpy.int(numpy.mean(nonzerox[right_within_window_indices]))

        return left_within_window_indices, right_within_window_indices


def _draw_polygon(y, left_x, right_x, shape):
    warp_zero = numpy.zeros(shape).astype(numpy.uint8)
    color_warp = numpy.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    left_points = numpy.array([numpy.transpose(numpy.vstack([left_x, y]))])
    right_points = numpy.array([numpy.flipud(numpy.transpose(numpy.vstack([right_x, y])))])
    pts = numpy.hstack((left_points, right_points))

    # Draw the lane onto the warped blank image
    return cv2.fillPoly(color_warp, numpy.int_([pts]), (0, 255, 0))


def _draw_lines(warped_binary, y, left_fit, right_fit, nonzeroy, nonzerox, left_indices, right_indices):
    out = numpy.dstack((warped_binary, warped_binary, warped_binary))
    out[nonzeroy[left_indices], nonzerox[left_indices]] = [255, 0, 0]
    out[nonzeroy[right_indices], nonzerox[right_indices]] = [0, 0, 255]

    leftfitx = left_fit.astype(numpy.int32)
    rightfix = right_fit.astype(numpy.int32)
    ycoord = y.astype(numpy.int32)

    cv2.polylines(out, array(list(zip(leftfitx, ycoord))).reshape(-1, 1, 2), True, (0, 255, 0), thickness=10)
    cv2.polylines(out, array(list(zip(rightfix, ycoord))).reshape(-1, 1, 2), True, (0, 255, 0), thickness=10)
    return out


def _put_text(image, distance_from_center, left_coeffs_m, right_coeffs_m):
    left_curverad = (1 + (2 * left_coeffs_m[0] * 720 * ym_per_pix + left_coeffs_m[1]) ** 2) ** 1.5 / \
                    numpy.absolute(2 * left_coeffs_m[0])

    right_curverad = (1 + (2 * right_coeffs_m[0] * 720 * ym_per_pix + right_coeffs_m[1]) ** 2) ** 1.5 / \
                     numpy.absolute(2 * right_coeffs_m[0])

    cv2.putText(image, 'left: {}'.format(left_curverad), position, font, fontScale, fontColor, lineType)

    cv2.putText(image, 'right: {}'.format(right_curverad), (position[0], position[1] + 40),
                font, fontScale, fontColor, lineType)

    cv2.putText(image, 'dist. to center: {}'.format(xm_per_pix * distance_from_center),
                (position[0], position[1] + 80), font, fontScale, fontColor, lineType)

    return image


class Pipeline:
    def __init__(self, model=None, debug=None):
        self._camera_matrix, self._distortion_coefs = get_calibration_results()
        self._binary_model = keras.models.load_model(model, custom_objects={'Upsampling': Upsampling})
        self._lines = Lines()
        self._debug = debug

    def __call__(self, image, **kwargs):
        undistorted = undistort(image, self._camera_matrix, self._distortion_coefs, None, None)
        warped = perspective_transform(undistorted, perspective_tr_matrix)

        if self._debug == 'perspective':
            return warped

        warped_binary = self._get_thresholded(warped)
        warped_binary_in_rgb = numpy.dstack((warped_binary, warped_binary, warped_binary)) * 255

        if self._debug == 'warped':
            return warped_binary_in_rgb

        y, left_fit, right_fit, nonzeroy, nonzerox, left_indices, right_indices = self._lines.get_lines(warped_binary)

        lines_drawn = _draw_lines(warped_binary,
                                  y,
                                  left_fit,
                                  right_fit,
                                  nonzeroy,
                                  nonzerox,
                                  left_indices,
                                  right_indices)

        if self._debug == 'lines':
            return lines_drawn

        polygon_drawn = _draw_polygon(y, left_fit, right_fit, warped_binary.shape)

        if self._debug == 'polygon':
            return polygon_drawn

        unwarped_polygon = perspective_transform(polygon_drawn, inverse_perspective_tr_matrix)
        main_track = cv2.addWeighted(undistorted, 1, unwarped_polygon, 0.5, 0)
        _put_text(main_track, (left_fit[-1] + right_fit[-1]) / 2 - columns / 2,
                  *self._lines.curve_coefficients_in_meters())

        additional = numpy.concatenate((warped, warped_binary_in_rgb, lines_drawn), axis=1)
        additional = cv2.resize(additional, (1280, 240))
        return numpy.concatenate((additional, main_track), axis=0)

    def _get_thresholded(self, image):
        result = self._binary_model.predict(image.reshape(1, *image.shape)).squeeze() * 255
        result = result.astype(numpy.uint8)
        result = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 0)
        return result // 255


if __name__ == '__main__':
    pass
