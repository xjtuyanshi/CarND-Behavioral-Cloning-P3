# **Behavioral Cloning** 
---
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./training_data/IMG/center_2019_01_06_23_44_30_983.jpg "Center Driving Image"
[image2]: ./training_data/IMG/center_2019_01_06_23_48_44_910.jpg "Recovery Image"
[image3]: ./training_data/IMG/center_2019_01_06_23_48_43_702.jpg "Recovery Image"
[image4]: ./training_data/IMG/center_2019_01_06_23_47_33_403.jpg "Recovery Image"
[image5]: ./augmented_image_examples/raw_image.png?raw=true "Normal Image"
[image6]: ./augmented_image_examples/Flipped_image.png?raw=true  "Flipped Image"
[image7]: ./augmented_image_examples/darker_image.png?raw=true  "Darked Image"
[image8]: ./hist_angles.png?raw=true  "Histogram of angles"
[image9]: ./model_training_validation_epoch.png?raw=true  "Train validation loss vs epoch"
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model_train.py containing the jupyter notebook model training methods exploration process
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* readme.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a NVIDIA convolution neural network (model.py lines 133-157) 

The model includes RELU layers to introduce nonlinearity (code line 139-154), and the data is normalized in the model using a Keras lambda layer (code line 138). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers after every convets layers in order to reduce overfitting (model.py lines 139-155). 

The model was trained and validated on different data sets to ensure that the model was not 
overfitting (code line 166). I let the keras to split the data to 80% training data and 20 % validation data
 The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 165).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 
I used a combination of:
* 3 laps of center lane driving
* 1 lap recovering from the left and right sides of the road
* 1 lap reverse driving
* + ~8000 Udacity training data

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach


My first step was to use a convolution neural network model similar to the NVIDIA model from 
their paper End to End Learning for Self-Driving Cars.
 I thought this model might be appropriate because their model is proved to very effective to model
 behavior cloning.

Below is the model structure:

|Layer (type)       |             Output Shape       |       Param #  |
|:---------------------:  |:--------------------------|-------------------:| 
|cropping2d_1 (Cropping2D) |   (None, 90, 320, 3)    |     0         |
| lambda (Lambda)          |    (None, 66, 200, 3)   |     0       |  
| lambda_1 (Lambda)        |    (None, 66, 200, 3)   |     0       |  
| conv2d (Conv2D)          |    (None, 31, 98, 24)    |    1824    |  
| dropout (Dropout)       |     (None, 31, 98, 24)    |    0       |  
|conv2d_1 (Conv2D)         |   (None, 14, 47, 36)     |   21636    | 
|dropout_1 (Dropout)       |   (None, 14, 47, 36)   |     0        | 
|conv2d_2 (Conv2D)        |    (None, 5, 22, 48)     |    43248    | 
|dropout_2 (Dropout)       |   (None, 5, 22, 48)     |    0         |
|conv2d_3 (Conv2D)        |   (None, 3, 20, 64)     |    27712     |
|dropout_3 (Dropout)      |   (None, 3, 20, 64)     |    0         |
|conv2d_4 (Conv2D)        |   (None, 1, 18, 64)     |    36928     |
|dropout_4 (Dropout)      |   (None, 1, 18, 64)     |    0         |
|flatten (Flatten)        |   (None, 1152)          |    0         |
|dense (Dense)            |   (None, 100)           |    115300    |
|dropout_5 (Dropout)       |  (None, 100)            |   0         |
|dense_1 (Dense)           |  (None, 50)             |   5050      |
|dropout_6 (Dropout)       |  (None, 50)             |   0         |
|dense_2 (Dense)           |  (None, 10)             |   510       |
|dropout_7 (Dropout)       |  (None, 10)             |   0         |
|dense_3 (Dense)           |  (None, 1)              |   11        |

The first 3 layers is for preprocessing the input data.
* Crop the images to exclude the sky and the driving car.
* Resize the image to 66X200 to reach the required input image size same as NVIDIA model
* Normalize images. <br/>

After that, the model is same as the model architecture described in NVIDIA's paper.
In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 
To combat the overfitting, I modified the model by adding dropout layer after every original layer
so that it randoms drops out some information and successfully prevent overfitting.


The final step was to run the simulator to see how well the car was driving around track one.
 There were a few spots where the vehicle fell off the track especially when the car is driving on the curves.
 To improve the driving behavior in these cases, I created several recovery data which tells the model how to turn back to the center from side of track.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.


#### 2. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back 
to center so that the vehicle would learn to how to recover from side of track if the 
car accidentally off the center of the track. These images show what a recovery looks like 
starting from edge of the track :


![alt text][image2]
![alt text][image3]
![alt text][image4]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would improve the robustness of model.
For example, here is an image that has then been flipped:  
![alt text][image5]  
![alt text][image6]  

Besides, I also decreased the brightness randomly for same images.
![alt text][image7]  
Lastly, I randomly removed 20% of center camera images to make the model not quite biased to center.


After the collection process, I had 48134 number of data points and the distribution of data is:

![alt text][image8]

I didn't use generators to avoid the memory issue as generator's training speed is way slower than
 than directly load all images to train. (Code for generator can be found in the model_train.ipynb)
I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 150 as evidenced by following chart.
I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image9]

The final video can be found in  **[YouTube](https://youtu.be/-O3C6sFyzTk)**