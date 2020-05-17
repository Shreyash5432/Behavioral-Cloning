# Behavioral-Cloning
The project is focussed on teaching a convolutional neural network to drive around a track without leaving the driving surface.

## CNN Model Architecture
My first step was to use a convolution neural network model similar to Nvidia’s End-to-End deep learning model.  I thought this model might be appropriate because it uses an end-to-end approach which means it uses minimum training data to learn to steer around the road.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model was overfitting. To combat overfitting, I added two dropout layers with keeping probability equal to 90%.

An additional change to Nvidia’s model is the depth of the fifth convolutional layer. The depth of the fifth convolutional layer in my model is 128 in contrast to the depth in Nvidia’s model which is 64. This change was done to add depth to the neural network which improves the feature extraction of the image. While training, the model uses Adam's optimizer with batch size and the number of epochs equal to 32 and 6 respectively.

The final model architecture along with the input and output shape of the image is shown in the image titled "Model Architecture".

<p align="center">
  <b>Model Architecture</b><br>
  <img src="https://static.wixstatic.com/media/bb5837_98f601f31a214836badf70093e142577~mv2.png/v1/crop/x_0,y_100,w_473,h_1620/fill/w_473,h_1620,al_c,q_95/model_plot.webp">
</p>

## Training Strategy
The model architecture was developed in a way that it can learn the steering angle accurately using the smaller dataset of images from the front camera. I recorded one lap of the track using center lane driving (see Video: Center lane driving) along with some maneuvers of vehicle recovering from the left and right side of the road back to the center (see Video: Recovering maneuver).

A single frame in the video gave three outputs: images from the camera mounted on the left, center and right side of the vehicle. To augment the center camera image data with left and right camera image data, I used a correction factor for the steering wheel. So, for the same frame, the steering angle for the left, center and right camera image will be:

Steering angle for the left camera image: (Center_Steering_angle + Correction_factor)

Steering angle for the center camera image: (Center_Steering_angle)

Steering angle for the right camera image: (Center_Steering_angle – Correction_factor)

Also, to further enlarge the dataset, the images were flipped in order to add driving behavior for the right turn as the track only had left turns. The steering angle for the flipped image will be:

Steering angle for the normal camera image: (Steering_angle)

Steering angle for the flipped camera image: (-Steering_angle)

After the collection process, I had total of 48516 images in the dataset. I then preprocessed this data by normalizing and cropping the images with the Keras Lambda layer and the Cropping2d layer respectively. Then I finally randomly shuffled the data set and put 20% of the recorded data into the validation set.

<p align="center">
  <b>Left Camera Image</b><br>
  <img src="https://static.wixstatic.com/media/bb5837_59b849a9ead640958b0ec9bbabf4afef~mv2.jpg/v1/fill/w_448,h_224,al_c,lg_1,q_90/left_2020_04_11_00_52_31_960.jpg">
</p>

<p align="center">
  <b>Center Camera Image</b><br>
  <img src="https://static.wixstatic.com/media/bb5837_3053c350b3914d21a22cf50ec3e4fb99~mv2.jpg/v1/fill/w_448,h_224,al_c,lg_1,q_90/center_2020_04_11_00_52_31_960.jpg">
</p>

<p align="center">
  <b>Right Camera Image</b><br>
  <img src="https://static.wixstatic.com/media/bb5837_8ac6e469b34c4de59f36175456d2d300~mv2.jpg/v1/fill/w_448,h_224,al_c,lg_1,q_90/right_2020_04_11_00_52_31_960.jpg">
</p>

<p align="center">
  <b>Flipped Image</b><br>
  <img src="https://static.wixstatic.com/media/bb5837_a8664db3ce91469da81e62040e71fb92~mv2.jpg/v1/fill/w_448,h_224,al_c,lg_1,q_90/image.jpg">
</p>

<p align="center">
  <b>Cropped Image</b><br>
  <img src="https://static.wixstatic.com/media/bb5837_ae2ce52a732f4db498fa44db25705024~mv2.png/v1/crop/x_0,y_55,w_320,h_85/fill/w_448,h_119,al_c,lg_1,q_95/center_2020_04_11_00_52_31_960.webp">
</p>
 
## Output: Testing the trained model
The video titled "Final Output" shows the result obtain while testing the Convolutional Neural Network model. We can see that the vehicle is driving around the race track without leaving the driving surface.
