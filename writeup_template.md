##Writeup Template
---

**Vehicle Detection Project**

The goal in this project is to detect and track the cars on the road. At least two different approaches can be used 
to spot the cars:
 * Using sliding window technique one can check many positions on an image for a car
 * Try to tell the coordinates of objects in an image directy (e.g, [YOLO](https://arxiv.org/abs/1506.02640))

Since the project rubric explicitly asks for the usage of sliding window and [HOG](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) features 
I have tried the first approach to solve this problem. I have never done image classification with manually created features before, therefore decided also to compare a classifier that utilizes
these features against Convolutional Neural Networks. Furthermore, one of my objectives was to make the processing of videos
as fast as possible, without doing complicated post-processing after classification of the patches of images. 

The steps of this project are the following:

* Create a binary classifier for images (car vs. not car) using manually created features .
* Implement a sliding window technique and use trained classifier to search for vehicles in images.
* Run a pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[cars]: ./output_images/car.png
[not_cars]: ./output_images/not_car.png
[cars_hog_channel0]: ./output_images/car:channel0.png
[cars_hog_channel1]: ./output_images/car:channel1.png
[not_cars_hog0]: ./output_images/not_car_hog:channel0.png
[not_cars_hog1]: ./output_images/not_car_hog:channel1.png
[test_images]: ./output_images/test_images.png
[test_images_hog0]: ./output_images/test_images:channel0.png
[test_images_hog1]: ./output_images/test_images:channel1.png
[sliding_windows]: /output_images/sliding_windows.png
[found_boxes]: /output_images/found_boxes.png
[heatmaps]: /output_images/heatmaps.png

[video1]: ./project_video.mp4

---

###Car vs Not Car

![cars]

![not_cars]

To decide if a given image depicts a car or not, we have to build a binary classifier. [Here](https://github.com/turangojayev/CarND-Vehicle-Detection/blob/2302085cf756083bbf7847a23803f91013652f12/training.py#L67) I extract
HOG features from images as well as color histograms, for making use of them in classification. I exploit HOG features obtained 
from luma and blue difference channels of image converted to [YCbCr](https://en.wikipedia.org/wiki/YCbCr) color space. 
I group gradients calculated over the 8x8 patches of the image into 9 orients and normalize the values for each 2x2 block. 
Since it is a utilization of the gradients of the intensity values for pixels, these features help to spot the shape of the objects
we are looking for. Below, HOG features are depicted both for car images and for the one not containing a car:

![cars_hog_channel0]

![cars_hog_channel1]

![not_cars_hog0]

![not_cars_hog0]

We can also check how HOG features look like for the whole images:

![test_images]

On luma channel of YCbCr color space

![test_images_hog0]

On blue difference channel ofYCbCr color space

![test_images_hog1]

Moreover, I deploy color histograms binned into 32 groups for each of the [HLS](https://en.wikipedia.org/wiki/HSL_and_HSV) color space channels.
As a classifier model, I selected linear SVM and trained it with a class weight of 10 for "not car" and 1 for "car", with L2 regularization. 
The reason of higher class weight for the "not car" is the uneven distribution of the classes in a real world scenario. The described approach 
 results in accuracy of 0.990427927928 for test dataset (0.99 precision and 0.99 recall for both of the classes). 
 
 Selected features are not a reflection of the performance of the classifier on the dataset. I found many different combinations that yielded
 similar results on these data, but performed much poorer on videos (many more false positives). Therefore, during the selection of the features 
 the stress was made on getting better results for the videos and on run time.
 
 Classifier could probably be made even better, but already with HOG features for 2 channels and color histograms, the processing time was far from my goal for the project.
 
        [MoviePy] >>>> Building video test_output-hog.mp4
        [MoviePy] Writing video test_output-hog.mp4
         97%|█████████▋| 38/39 [00:10<00:00,  3.46it/s]
        [MoviePy] Done.
        [MoviePy] >>>> Video ready: test_output-hog.mp4 
 

        [MoviePy] >>>> Building video project_output-hog.mp4
        [MoviePy] Writing video project_output-hog.mp4
        100%|█████████▉| 1260/1261 [06:08<00:00,  3.47it/s]
        [MoviePy] Done.
        [MoviePy] >>>> Video ready: project_output-hog.mp4 


[Here](https://github.com/turangojayev/CarND-Vehicle-Detection/blob/2302085cf756083bbf7847a23803f91013652f12/training.py#L94)
is the procedure for the training of the model.

###Sliding Window Search
 Once we have a binary classifier, we can go over many positions on an image and try to spot a car. Since the cars can appear at different
 distances from the camera, care should be taken for different sizes corresponding to these distances. 
 
![sliding_windows]


---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

