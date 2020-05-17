### Importing Necessary Libraries ###

import csv
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

### Reading the csv file containing the patch of the images, steering and few other parameters ###

lines = []
with open('Data/driving_log.csv') as csvfile:
    Reader = csv.reader(csvfile)
    for line in Reader:
        lines.append(line)

### Storing the images and their steering measurements into arrays ###

images = []
measurements = []
correction_factor = 0.2
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = 'Data/IMG/' + filename
        image = cv2. imread(current_path)
        images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(measurement+correction_factor)
    measurements.append(measurement-correction_factor)

### Augmenting the recorded data with preprocessed data ###

aug_images = []
aug_measurements = []

for image, measurement in zip(images, measurements):
    aug_images.append(image)
    aug_measurements.append(measurement)
    flipped_image = cv2.flip(image,1)
    flipped_measurement = float(measurement) * (-1.0)
    aug_images.append(flipped_image)
    aug_measurements.append(flipped_measurement)

X_train = np.array(aug_images)
y_train = np.array(aug_measurements)

### Model Architecture ###

model = Sequential()

# Preprocessing the images - Normalizing image
model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(160,320,3)))

# Cropping the unnecessary part of an image
model.add(Cropping2D(cropping=((70,25),(0,0))))

# Layer 1: Convolutional Layer
model.add(Convolution2D(24,5,5,subsample = (2,2),activation='relu'))

# Layer 2: Dropout Layer
model.add(Dropout(0.1))

# Layer 3: Convolutional Layer
model.add(Convolution2D(36,5,5,subsample = (2,2),activation='relu'))

# Layer 4: Dropout Layer
model.add(Dropout(0.1))

# Layer 5: Convolutional Layer
model.add(Convolution2D(48,5,5,subsample = (2,2), activation='relu'))

# Layer 6: Convolutional Layer
model.add(Convolution2D(64,3,3, activation='relu'))

# Layer 7: Convolutional Layer
model.add(Convolution2D(128,3,3, activation='relu'))

# Flattening Layer
model.add(Flatten())

# Layer 8: Fully Connected Layer
model.add(Dense(1164))

# Layer 9: Fully Connected Layer
model.add(Dense(100))

# Layer 10: Fully Connected Layer
model.add(Dense(50))

# Layer 11: Fully Connected Layer
model.add(Dense(10))

# Layer 12: Fully Connected Layer
model.add(Dense(1))

### Training the model using Adam's optimizer ###
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 6)

### Saving the model ###
model.save('model.h5')
print('Model Saved')
