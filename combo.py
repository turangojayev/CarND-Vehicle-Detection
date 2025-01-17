import glob

import cv2
import keras
import numpy
import tensorflow as tf
from keras.layers import *

from vehicle_detection import _draw_boxes_on, process_and_save_video, CNNVehicleDetector, search_parameter

COLUMNS = 9
ROWS = 6
rows = 720
columns = 1280

src = numpy.float32([[0, 700],
                     [515, 472],
                     [764, 472.],
                     [1280, 700.]])

dst = numpy.float32([[100, 710],
                     [100, 10],
                     [1180, 10],
                     [1180, 710]])

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


class Pipeline:
    def __init__(self, vehicle_detector, lane_line_detector):
        self._camera_matrix, self._distortion_coefs = get_calibration_results()
        self._vehicle_detector = vehicle_detector
        self._lane_line_detector = lane_line_detector

    def __call__(self, image):
        image = undistort(image, self._camera_matrix, self._distortion_coefs, None, None)

        boxes = self._vehicle_detector(image)
        if isinstance(boxes, numpy.ndarray):
            raise TypeError('List of tuples was expected, but instead got \'numpy.ndarray\', '
                            'instantiate vehicle_detector with return_boundaries=True')

        unwarped = self._lane_line_detector(image)
        image = cv2.addWeighted(image, 1, unwarped, 0.5, 0)
        return _draw_boxes_on(boxes, image)


class LaneLineDetector:
    def __init__(self):
        self._lines = Lines()

    def __call__(self, image):
        warped = perspective_transform(image, perspective_tr_matrix)
        warped_binary = self._get_thresholded(warped)
        y, left_fit, right_fit, *_ = self._lines.get_lines(warped_binary)
        polygon_drawn = _draw_polygon(y, left_fit, right_fit, warped_binary.shape)
        return perspective_transform(polygon_drawn, inverse_perspective_tr_matrix)

    def _get_thresholded(self, image):
        pass


class SegmentationBasedLineDetector(LaneLineDetector):
    def __init__(self, model):
        super(SegmentationBasedLineDetector, self).__init__()
        self._model = model

    def _get_thresholded(self, image):
        result = self._model.predict(image.reshape(1, *image.shape)).squeeze() * 255
        result = result.astype(numpy.uint8)
        result = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 0)
        return result // 255


class IntensityThresholdingLineDetector(LaneLineDetector):
    def __init__(self):
        super(IntensityThresholdingLineDetector, self).__init__()

    def _get_thresholded(self, image):
        yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        yellow_color = cv2.inRange(yuv[:, :, 1], array([0]), array([115]))
        yellow_color[yellow_color != 0] = 1

        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        white_color = cv2.inRange(hls, array([0, 200, 0]), array([255, 255, 255]))
        white_color[white_color != 0] = 1

        out = numpy.zeros_like(white_color)
        out[(white_color != 0) | (yellow_color != 0)] = 1
        return out


if __name__ == '__main__':
    process_and_save_video('project_video.mp4', 'output_videos/combo-cnn.mp4',
                           Pipeline(
                               CNNVehicleDetector(
                                   vehicle_classifier=keras.models.load_model('car_models/model-28-0.995.h5'),
                                   search_params=[
                                       search_parameter(1.5, 0.5, 0.5, 400, 464),
                                       search_parameter(2, 0.5, 0.5, 400, 592),
                                   ],
                                   weight_for_new_frame=0.3,
                                   return_boundaries=True
                               ),
                               # SegmentationBasedLineDetector(
                               #     keras.models.load_model('car_models/model.h5',
                               #                             custom_objects={'Upsampling': Upsampling}))
                               IntensityThresholdingLineDetector()
                           ))
