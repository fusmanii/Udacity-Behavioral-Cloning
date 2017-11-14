#**Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

[architecture]: https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png "nVidia model"
[finalVideo]: ./writeup_data/final_run.gif "Final Run Video"
[recoveryVideo]: ./writeup_data/recover.gif "Recovery Video"

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_data/good_driving-center.jpg "Good Driving Center"
[image2]: ./writeup_data/recover_center1.jpg "Recover Center"
[image3]: ./writeup_data/recover_center2.jpg "Recover Center"
[image4]: ./writeup_data/recover_center3.jpg "Recover Center"
[image5]: ./writeup_data/recover_center4.jpg "Recover Center"
[image6]: ./writeup_data/recover_center5.jpg "Recover Center"
[image7]: ./writeup_data/1.png "Distribution before"
[image8]: ./writeup_data/2.png "Distribution after"
[image9]: ./writeup_data/3.png "Flip before"
[image10]: ./writeup_data/4.jpg "Flip after"
[image11]: ./writeup_data/5.png "Processed image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.json
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network that consists of 3 5x5 convolution layers alternating with ELU layers (code line 157-162).

Then followed by 2 3x3 convolution layers alternating with ELU layers (code line 165-168).

Finally, there are 3 fully connected layers (code line 174-179).

The data is normalized in the model using a Keras lambda layer (code line 154). 

#### 2. Attempts to reduce overfitting in the model

The data was augmented and to reduce overfitting random distrtion was introduced to the new images like random brightness or shift (code line 70-102).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 186-189). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 184).

#### 4. Appropriate training data

For the training data the car was driven on the road. All three, center, left and right lane images were used. Additionaly, crash recovery was added to the data set to improve driving proformace 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach


The overall strategy was to first obtain data from the driving simulation. Then augment the data as to make the angle distribution more even and also add right turns by fliping the images and angles horizontally.

My first step was to use a convolution neural network model similar to the model that was proposed by the instructor. I thought this model might be appropriate because it was used by nVidia to drive a real car.

I split my image and steering angle data into a training and validation set. 

To combat the overfitting, I modified the model where I added dropout with keep probability of 0.5 and L2 regularization with lambda of 0.001. Later I noticed that the dropout seem wasn't improving performace for I removed all the dropouts.

Two useful sugestions were adapted from the nanodegree community: 

* Using `ELU` activation instead of `ReLU` that originally used
* Using keras ModelCheckpoint to save the model after each epoch

The final step was to run the simulator to see how well the car was driving around track one. The car was driving with in the tracks but was bouncing back and forth between the side lines. However, the bouncing back and forth was reduced once I implmented the two sugestions form nanodegree community that I mentioned above.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

![finalVideo]

#### 2. Final Model Architecture

The final model architecture (model.py lines 146-176) consisted of a convolution neural network with the following layers and layer sizes:

* Convolution layer: 5x5 kernel, 2x2 stride, 24 depth, L2 0.001
* ELU
* Convolution layer: 5x5 kernel, 2x2 stride, 36 depth, L2 0.001
* ELU
* Convolution layer: 5x5 kernel, 2x2 stride, 48 depth, L2 0.001
* ELU

* Convolution layer: 3x3 kernel, 64 depth, L2 0.001
* ELU
* Convolution layer: 3x3 kernel, 64 depth, L2 0.001
* ELU

* Flatten

* Fully connected layer: 100 depth, L2 0.001
* ELU
* Fully connected layer: 50 depth, L2 0.001
* ELU
* Fully connected layer: 10 depth, L2 0.001 
* ELU

* Fully connected layer: 1 depth

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![architecture]

#### 3. Creation of the Training Set & Training Process

Udacity provides the training data, but we are incuraged to collect additional data ourselfs. 

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover if the car were to ever go off the tracks. These images show what a recovery looks like starting from the car out side of the tracks and merging back to center of the track:

![image2]
![image3]
![image4]
![image5]
![image6]

Here is the video demonstrating the recovery:

![recoveryVideo]

Before augmenting or training I found the distribution of angles in the training data. This is what the distribution looked like:
![image7]

To have a more even distribution I removed angles (and assosiated images) with count above average.
This is what the distrubution looks like after:
![image8]

To avoid overfitting I augmented the data by flipping the images and angles horizontally to introduce right turns. For example, here is an image that has then been flipped:

![image10]
![image9]

After the collection process, I had 5387 number of data points. I then preprocessed this data by croping the image above the track, adding random brightness and random shift. Here what the final image looks like.
![image11]


I finally randomly shuffled the data set and put 5% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the validation loss stop decreasing after the fifth epoch. I used an adam optimizer so that manually training the learning rate wasn't necessary.
