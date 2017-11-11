##Writeup Template
---

**Vehicle Detection Project**

The goal in this project is to detect and track the cars on the road. At least two different approaches can be used 
to spot the cars:
 * Using sliding window technique one can check many positions on an image for a car
 * Try to tell the coordinates of objects in an image directy (e.g, [YOLO](https://arxiv.org/abs/1506.02640))

Since the project rubric explicitly asks for the usage of sliding window and [HOG](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) features 
I have tried the first approach to solve this problem. I have never done image classification with manually created features before, therefore decided also to compare a classifier that utilizes
these features against Convolutional Neural Networks.

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

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
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

