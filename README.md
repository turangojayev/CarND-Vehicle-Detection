**Writeup Template**
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
[result]: /output_images/result.png

[video1]: output_videos/project_output-hog.mp4
[video2]: /output_videos/test_output-hog.mp4

---

**Car vs Not Car**

![cars]

![not_cars]

To decide if a given image depicts a car or not, we have to build a binary classifier. 
[Here](https://github.com/turangojayev/CarND-Vehicle-Detection/blob/e424f77f41c3d056683b1e77fc6a7f1978864a65/training.py#L67) I extract
HOG features from images as well as color histograms, for making use of them in classification. I exploit HOG features obtained 
from luma and blue difference channels of image converted to [YCbCr](https://en.wikipedia.org/wiki/YCbCr) color space. 
I group gradients calculated over the 8x8 patches of the image into 9 orients and normalize the values for each 2x2 block. 
Since it is a utilization of the gradients of the intensity values for pixels, these features help to spot the shape of the objects
we are looking for. Below, HOG features are depicted both for car images and for the one not containing a car:

![cars_hog_channel0]

![cars_hog_channel1]

![not_cars_hog0]

![not_cars_hog1]

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


[Here](https://github.com/turangojayev/CarND-Vehicle-Detection/blob/e424f77f41c3d056683b1e77fc6a7f1978864a65/training.py#L94)
is the procedure for the training of the model.

**Sliding Window Search**

 Once we have a binary classifier, we can go over many positions on an image and try to spot a car. Since the cars can appear at different
 distances from the camera, care should be taken for different sizes corresponding to these distances. I have used sliding window approach for
  the scales of 1.5 and 2 (in pixesl 96 and 128 respectively). For both scales, I have made use of the rectangle area between 
  with a height of 256 (400-656 of height) and used overlap of 75% between subsequent windows. The exploited area is plotted below:
 
![sliding_windows]

Here is an example of the results of the classification for windows:

![found_boxes]

---

**Video Implementation**

As we can see from the image above, several overlapping windows can be classified as containing a car. For these cases as 
well as for removing false positives, heatmaps can give a better representation:

![heatmaps]

I took a time for coming with a solution that finds minimal number of false positives. To keep the logic of post-processing
as simple as possible I average the heatmaps from last 10 frames and apply a threshold of 0.6. Code for this process can be found
[here](https://github.com/turangojayev/CarND-Vehicle-Detection/blob/fa0c3a71dc7010e0cef42bef193adbe9d2f86e80/vehicle_detection.py#L230). Afterwards,
I use `scipy.ndimage.measurements.label()` to find blobs in the heatmap and assuming that they correspond to cars, I draw a
rectangle around each of them:

![result]


The resulting videos can be found below: 

[Project video](https://github.com/turangojayev/CarND-Vehicle-Detection/blob/master/output_videos/project_output-hog.mp4)

[Test video](https://github.com/turangojayev/CarND-Vehicle-Detection/blob/master/output_videos/test_output-hog.mp4)

---

**Too slow, what to do?**

I have spent a lot of time trying to make SVM classifier work well on video file. Accuracy, recall and precision of the classifier are not sufficient
to judge how it will perform on videos. Despite very high results on datasets(no overfitting!), the models kept performing poorly on video. I started with only HOG features and only for one channel, which was performing around 15 fps, however, finding a lot of false positives. By introducing more features,
of course, speed was damaged (reduced to 3-4 fps). To overcome this problem, as well as to compare learning features from images with neural network against manually
 created ones, I used [very simple CNN](https://github.com/turangojayev/CarND-Vehicle-Detection/blob/9abd7f3c3ad342aff4a47eb9008ec27ca5b0d20c/with_nn.py#L25) consisting of two convolutional and two fully connected layers before the output layer as a classifier.
 It runs much quicker(real-time) and results in almost zero false positives. To remove rest of them and to make the pipeline to
 show the car position also on frames, where a car was not detected, I calculate all the sliding window positions ahead of time and 
 use running average of the probabilities for the window positions ([here](https://github.com/turangojayev/CarND-Vehicle-Detection/blob/ec650d6e3162bd6e8fa0c1c767a060a1e899502f/vehicle_detection.py#L272)).
 
    [MoviePy] >>>> Building video test_output-cnn.mp4
    [MoviePy] Writing video test_output-cnn.mp4
     77%|███████▋  | 30/39 [00:00<00:00, 47.42it/s] 90%|████████▉ | 35/39 [00:00<00:00, 47.48it/s] 97%|█████████▋| 38/39 [00:00<00:00, 47.60it/s]
    [MoviePy] Done.
    [MoviePy] >>>> Video ready: test_output-cnn.mp4 
    [MoviePy] >>>> Building video project_output-cnn.mp4
    [MoviePy] Writing video project_output-cnn.mp4
     99%|█████████▉| 1253/1261 [00:26<00:00, 46.22it/s]100%|█████████▉| 1258/1261 [00:27<00:00, 47.21it/s]100%|█████████▉| 1260/1261 [00:27<00:00, 46.44it/s]
    [MoviePy] Done.
    [MoviePy] >>>> Video ready: project_output-cnn.mp4 

Here are the links two outputs made by CNN:

[Project video](https://github.com/turangojayev/CarND-Vehicle-Detection/blob/master/output_videos/project_output-cnn.mp4)

[Test video](https://github.com/turangojayev/CarND-Vehicle-Detection/blob/master/output_videos/test_output-cnn.mp4)

[combo]: ./output_images/hqdefault.jpg
I also [combined](https://github.com/turangojayev/CarND-Vehicle-Detection/blob/master/combo.py) the pipeline for vehicle detection with a lane-line detection pipeline, and the resulting link on youtube is below 

[![combo]](https://www.youtube.com/watch?v=nemEiZ-F5tM&feature=youtu.be)

That being said, there are still lots of problems with sliding windows approach. It requires many steps over one frame,
what reduces the speed of processing. Moreover, several scales for windows should be selected to make detection smoother (which will again drop the speed). If manually constructed features are being used, it results in false positives as well as further speed reduce. Although I have used CNNs to do binary classification with sliding window
 approach, it can be replaced by an approach like YOLO, where each frame is fed to neural network once and coordinates found directly. 