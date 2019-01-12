import csv
import os
import cv2
import datetime
import matplotlib.image as mpimg
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Dropout,Activation,Flatten,Lambda,Cropping2D,Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import time
import random
import gc
from tensorflow.python.keras.callbacks import ModelCheckpoint
samples = []

# read udacity images
with open(os.path.join('data', 'driving_log.csv')) as csvfile:
    reader = csv.reader(csvfile)
    # Skip header row.
    next(reader)
    for line in reader:
        samples.append(line)

# read my training data
with open(os.path.join('training_data', 'driving_log.csv')) as csvfile:
    reader = csv.reader(csvfile)
    # Skip header row.
    next(reader)
    for line in reader:
        samples.append(line)

#--------------helper functions for reading and resize images-------------------------
def read_img(path):
    if (path.__contains__('training_data')):
        filename = path.split('\\')[-1]
        file_path = os.path.join('training_data', 'IMG', filename)
    else:
        filename = path.split("/")[-1]
        file_path = os.path.join('data', 'IMG', filename)
    image = mpimg.imread(os.path.join(file_path))
    return image

def resize_img(image):
    resized = tf.image.resize_images(image, (66, 200))
    return resized

#----------------------Data augmentation helper fuuntions-----------------------------
def decrease_brightness(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    rand_num = random.uniform(0.6, 0.9)
    img_hsv[:, :, 2] = rand_num * img_hsv[:, :, 2]
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

def filp_image_angle(img, angle):
    flipped_img = cv2.flip(img, 1)
    filpped_angle = angle * (-1.0)
    return flipped_img, filpped_angle

# load the images and angles
def load_data(samples):
    images = []
    angles = []
    # in order to avoid bias of center angles, we random pick 80% center images with 0 angle
    center_image_keep_rate = 0.8
    # angle correction if we use right or left images
    correction = 0.23

    for sample in samples:
        center_image = read_img(sample[0])
        center_angle = float(sample[3])
        if np.random.random() < center_image_keep_rate:
            images.append(center_image)
            angles.append(center_angle)
            # if this steel wheel turns add right left camerea images
        if center_angle != 0:
            left_image = read_img(sample[1])
            left_angle = center_angle + correction
            right_image = read_img(sample[2])
            right_angle = center_angle - correction
            images.append(left_image)
            images.append(right_image)
            angles.append(left_angle)
            angles.append(right_angle)

    return images, angles


#----------------------implement data augmentation-----------------------------
def random_augment(images, angles):
    random_decrease_brightness_rate = 0.3
    augmented_images = images.copy()
    augmented_angles = angles.copy()

    for image, angle in zip(images, angles):

        if angle != 0:
            flipped_image, flipped_angle = filp_image_angle(image, angle)
            augmented_images.append(flipped_image)
            augmented_angles.append(flipped_angle)
        # random_decrease_brightness
        if  np.random.random() < random_decrease_brightness_rate:
            dark_image = decrease_brightness(image)
            dark_image_angle = angle
            augmented_images.append(dark_image)
            augmented_angles.append(dark_image_angle)
    X = np.array(augmented_images)
    y = np.array(augmented_angles)
    return shuffle(X, y)

#-----------------load raw images and augmented images-----------------------
t0 = time.time()
raw_images,raw_angles = load_data(samples)
print("raw images loaded")
X_train,y_train =random_augment(raw_images,raw_angles)
print("augmented images loaded")
del raw_angles,raw_images
gc.collect()
t1=time.time()
print("total seconds for loading and agumenting: {} sec".format(round(t1-t0)))
print('augmented images size:' +str(X_train.shape[0]) +"augmented angle size:" + str(y_train.shape[0]))

# print histogram of angles
num_bins=50
n, bins, patches = plt.hist(y_train, num_bins, facecolor='blue', alpha=0.5)
plt.show()

# hyper prameters
N_EPOCHS = 512
BATCH_SIZE = 1024
learning_rate = 0.0001

##NVIDIA MODEL
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Cropping2D(cropping=((45,25),(0,0)),input_shape=(160,320,3)))
model.add(Lambda(resize_img))
model.add(Lambda(lambda x:x/255-0.5))
model.add(Conv2D(24,(5,5),strides=(2,2),activation='relu'))
model.add(Dropout(.5))
model.add(Conv2D(36,(5,5),strides=(2,2),activation='relu'))
model.add(Dropout(.5))
model.add(Conv2D(48,(5,5),strides=(2,2),activation='relu'))
model.add(Dropout(.5))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Dropout(.5))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Dropout(.5))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(10, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(1))
model.summary()

# run model
checkpoint = ModelCheckpoint('model-{val_loss:03f}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')
model.compile(loss='mse', optimizer='Adam')
history_object = model.fit(X_train, y_train,validation_split =0.2,batch_size= BATCH_SIZE,shuffle=True,epochs=N_EPOCHS,callbacks=[checkpoint],verbose = 1)
model_name = 'model_NVIDIA_'+datetime.datetime.now().strftime("%Y%m%d%H%M")+'.h5'
model.save(model_name)
print("DONE!")