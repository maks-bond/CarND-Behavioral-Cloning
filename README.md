# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road

[//]: # (Image References)

[image1]: ./writeup_imgs/img1.png "Fail 1"
[image2]: ./writeup_imgs/img2.png "Fail 2"
[image3]: ./writeup_imgs/loss.jpg "Loss"

### Data collection
The data has been collected on two tracks using Udacity Simulator. The goal of the project is to drive the car autonomously on the first track only, which is simpler than second track. However data from two tracks has been collected to better generalize the network and also an attempt to drive the car autonomously on the second track has been performed as well (see results below).

### Model Architecture and Training Strategy

#### Architecture
The sequential convolutional neural network has been built following the idea presented in the following blogpost by Nvidia:  https://devblogs.nvidia.com/deep-learning-self-driving-cars/  

Network receives RGB image as an input and outputs steering angle.  
The idea is to have convolutional layers which produce 'fatter' but 'smaller' outputs (the number of filters increases but 'height' and 'width' decrease).
Bottom layers are all dense layers with reducing number of neurons.

Here is the Keras summary of the model:  

**Layer (type)** |                **Output Shape**  |            **Param #**   |
|---|---|---|
lambda_1 (Lambda)        |    (None, 160, 320, 3)    |   0  |   
cropping2d_1 (Cropping2D)  |  (None, 80, 320, 3)      |  0  |      
conv2d_1 (Conv2D)       |     (None, 38, 158, 24)   |    1824     
batch_normalization_1 (Batch) | (None, 38, 158, 24)  |     96       
dropout_1 (Dropout)         | (None, 38, 158, 24)     |  0        
conv2d_2 (Conv2D)           | (None, 17, 77, 36)    |    21636    
batch_normalization_2 (Batch) | (None, 17, 77, 36)   |     144      
dropout_2 (Dropout)         | (None, 17, 77, 36)     |   0        
conv2d_3 (Conv2D)            | (None, 7, 37, 48)     |    43248    
batch_normalization_3 (Batch) | (None, 7, 37, 48)    |     192      
dropout_3 (Dropout)         | (None, 7, 37, 48)   |      0        
conv2d_4 (Conv2D)         |   (None, 3, 18, 64)    |     27712    
conv2d_5 (Conv2D)       |     (None, 1, 8, 64)      |    36928    
flatten_1 (Flatten)    |      (None, 512)      |         0        
dense_1 (Dense)     |         (None, 100)       |        51300    
dense_2 (Dense)      |        (None, 50)         |       5050     
dense_3 (Dense)       |       (None, 10)          |      510      
dense_4 (Dense)        |      (None, 1)            |     11            


#### Overfitting
Initially I observed that the training loss is much smaller than validation loss. Also training loss has been reducing with each epoch but validation loss has remained the same or even was growing. Which indicated overfitting.  
Addition of some dropout layers seems to have helped with overfitting problem. We have three dropout layers after first three conv layers with 0.5 probability of ignoring the connection.  
Also colleccting more data has helped with overfitiing problem as well.   
Originally I've been training the network on around 5,000 images but in the end I had around 50,000 images which improved the behavior of autonomous driving and helped with overfitting.

#### Training data  
Data from first and second tracks of the following types of driving has been collected:  
1. Careful driving at the center of the lane for several laps
2. Repeated driving near complex turns
3. Recovery driving started at the position where the car is about to go off the road and managing to stay on the road
4. Driving in the opposite direction of the track to better generalize the model.

#### Data preprocessing
Each image has been flipped to enhance the dataset. Also the steering angle value is negated for flipped image. Each image is normalized to have 0 mean and standard deviation of 1 and cropped to remove the top 'sky' area and bottom 'hood' area.  
Note that Udacity Simulator additionally provides left and right images produced from the left and right corners of the car hood. I decided to not use them to eliminate experimentaion with steering angle offset. Instead I just collected more data of different driving conditions.

#### Model evolution
1. I have started with the simple model of two conv layers and max pooling layers in between. After that I had flatten and 2 dense layers. Such model is similar to LeNet architecture for digit image classification. Also I have collected around 3000 training data images. Such model was autonomously driving the car well before first complex turn on track 1 where the car was jut going straight:  

![alt text][image1]

2. I collected more data around this turn including recovery data to 'show' the model how to behave if car is almost out of the road. Also the model has been improved to be like Nvidia architecture described above. That helped and the car has started making a turn at that point. However it was failing at the next complex turn:  
![alt text][image2]  

Instead of making a turn it was just going straight into the 'shortcut' area.

3. Again such problem was solved with more data collection of careful driving especially near that turn, recovery data and driving in the opposite direction. More data and only 5 epochs of training allowed to achieve success on track 1. Validation loss was not much higher than training loss which indicated lack of overfitting.  

4. The data from track 2 was collected for the purpose of generalization of the model and the attempt to autonomously drive on the track 2. The training and validation loss have been much higher with data from both tracks which is expected as second track is much more challenging and my driving there was not perfect which made it much harder for the model to learn my style of driving :-)  
 However new model still gave good results of autonomous driving on track 1 and not bad results of driving on track 2. At some point on track 2 the car goes off the road but in general I am very satisfied with the behavior of the model. The videos of autonomous driving on both tracks are linked below.  

5. After step 4 the model has performed relatively well at test time. However there was a gap between validation and training loss. At this step after each convulutional layer before dropout layer the batch normalization layer was added which is believed to help the model to better generalize. It resulted in the validation loss being much closer to training loss as you can see below on the plot without losing the quality of the produced model.

#### Training and validation loss visualization
In the end I trained the model for 10 epochs to observe the dynamic of training and validation losses:  
![alt text][image3]

As we can see the training loss always goes down which is expected and validation loss slowly but also goes down and is very close to the training loss. That shows that we do not have overfitting or underfitting problem, the loss is relatively small and model training process causes both training and validation losses to go down. Potentially we could train the model for even more number of epochs to observe the point where validation loss stops going down and plateaus or goes up.   

#### Autonomous driving videos
Track 1: https://www.youtube.com/watch?v=EmIwZbqW0EU   
Track 2: https://www.youtube.com/watch?v=VpaVv2o1LFQ 

The car drives through track 1 perfectly without touching edges of the road and almost always driving in the center of the lane. The turns are smooth.

The car is able to drive on the complex track 2 for some time before going off the road and not being able to recover.

#### Conclusion 
By collecting around 50,000 training images from both track 1 and track 2 and using Nvidia neural network architecture with some added dropout and batch normalization layers it was possible to train the model which can be used to autonomously drive the car in both tracks.  
The car fails at some point in the track 2 and potentially collecting more training data or improving the model could help to build even better model which would be robust on both tracks.  

Also our model uses just one image to predict steering angle independently of past images and predictions. It would be great to build a recurrent neural network which would take as an input previous steering angle. That would potentially produce much better model as we have high dependence between neighboring frames and next steering angle prediction should be very similar to previous one.