import math
import numpy as np        
import matplotlib.pyplot as plt

from os import getcwd
import csv
from PIL import Image         
import cv2 

import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, Callback
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.advanced_activations import ELU  

def generateTrainingData(imgPaths, angles, batchSize=128, validationFlag=False):
    '''
    Generator for training data.
    '''
    imgPaths, angles = shuffle(imgPaths, angles)
    X_data = []
    y_data = []
    while True:       
        for i in range(len(angles)):
            image = preprocess(cv2.imread(imgPaths[i]))
            angle = angles[i]

            if not validationFlag: 
                image = distortImage(image)
                
            # flip only images that have high enough angle
            if abs(angle) > 0.25:
                X_data.append(cv2.flip(image, 1))
                y_data.append(angle * -1)
                if len(X_data) == batchSize:
                    yield (np.array(X_data), np.array(y_data))
                    X_data = []
                    y_data = []
                    imgPaths, angles = shuffle(imgPaths, angles)

            X_data.append(image)
            y_data.append(angle)
            if len(X_data) == batchSize:
                yield (np.array(X_data), np.array(y_data))
                X_data = []
                y_data = []
                imgPaths, angles = shuffle(imgPaths, angles)

def preprocess(image):
    '''
    This method is simillar to the one in drive.py.
    '''
    return cv2.cvtColor(
        cv2.resize(
            cv2.GaussianBlur(
                image[50:140,:,:], 
                (3,3), 
                0
            ),
            (200, 66), 
            interpolation = cv2.INTER_AREA
        ), 
        cv2.COLOR_BGR2YUV
    )

def distortImage(image):
    ''' 
    Distort the image for generating training data to avoid overfitting.
    '''
    newImage = image.astype(float)
    
    # change the brightness by random amount
    amount = np.random.randint(-25, 25)
    newImage[:,:,0] += np.where(
        ((newImage[:,:,0] + amount) > 255) if amount > 0 else ((newImage[:,:,0] + amount) < 0), 
        0, 
        amount
    )
   
    # add random shift in the image
    h,w,_ = newImage.shape
    horizon = 2*h/5
    v_shift = np.random.randint(-h/8,h/8)
    newImage = cv2.warpPerspective(
        newImage,
        cv2.getPerspectiveTransform(
            np.float32([[0,horizon],[w,horizon],[0,h],[w,h]]),
            np.float32([[0,horizon+v_shift],[w,horizon+v_shift],[0,h],[w,h]])
        ),
        (w,h), 
        borderMode=cv2.BORDER_REPLICATE
    )
    return newImage.astype(np.uint8)

# Get the training data
imgPaths = []
angles = []

# pick my_data or udacity_data
pathPrepend = getcwd() + '/my_data/'
csvPath = './my_data/driving_log.csv'

with open(csvPath, newline='') as f:
    csvData = list(csv.reader(f, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))

csvData = csvData[1:]
for line in csvData:

    # ignore slow data
    if float(line[6]) < 0.1 :
        continue

    # get all three images: center, left and right
    imgPaths.append(pathPrepend + line[0])
    angles.append(float(line[3]))
    imgPaths.append(pathPrepend + line[1])
    angles.append(float(line[3])+0.25)
    imgPaths.append(pathPrepend + line[2])
    angles.append(float(line[3])-0.25)

imgPaths = np.array(imgPaths)
angles = np.array(angles)

# images for writeup
# image = cv2.imread(imgPaths[0])
# angle = angles[0]
# # image = preprocess(image)
# image = cv2.flip(image, 1)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# # image, angle = distortImage(image)
# print(image.squeeze().shape)
# plt.imshow(image)
# plt.show()

# histogram for writeup
numBins = 25
avgSamplesPerBin = len(angles)/numBins
hist, bins = np.histogram(angles, numBins)
width = 0.8 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
# plt.bar(center, hist, align='center', width=width)
# plt.plot((np.min(angles), np.max(angles)), (avgSamplesPerBin, avgSamplesPerBin), 'k-')
# plt.show()

# reduce the angles that show up a lot
target = avgSamplesPerBin * .5
keepProbs = [1.0 if hist[i] < target else 1./(hist[i]/target) for i in range(numBins)]
removeIndices = [i for i in range(len(angles)) for j in range(numBins) if angles[i] > bins[j] and angles[i] <= bins[j+1] and np.random.rand() > keepProbs[j]]
imgPaths = np.delete(imgPaths, removeIndices, axis=0)
angles = np.delete(angles, removeIndices)

# histogram again for writeup
hist, bins = np.histogram(angles, numBins)
# plt.bar(center, hist, align='center', width=width)
# plt.plot((np.min(angles), np.max(angles)), (avgSamplesPerBin, avgSamplesPerBin), 'k-')
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(imgPaths, angles, test_size=0.05, random_state=42)

print('NUM: ', len(X_train), len(X_test))
model = Sequential()

# Convolutional Neural Netwrok

# Normalize
model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(66,200,3)))

# 5x5 convolution layers (output depth 24, 36, and 48), each with 2x2 stride
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())

# 3x3 convolution layers (output depth 64, and 64)
model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())

# Flatten layer
model.add(Flatten())

# Add three fully connected layers (depth 100, 50, 10), tanh activation (and dropouts)
model.add(Dense(100, W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Dense(50, W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Dense(10, W_regularizer=l2(0.001)))
model.add(ELU())

# Fully connected output layer
model.add(Dense(1))

model.compile(optimizer=Adam(lr=1e-4), loss='mse')

trainGenerator = generateTrainingData(
    X_train, 
    y_train, 
    validationFlag=False, 
    batchSize=64
)
validGenerator = generateTrainingData(
    X_train, 
    y_train, 
    validationFlag=True, 
    batchSize=64
)
testGenerator = generateTrainingData(
    X_test, 
    y_test, 
    validationFlag=True, 
    batchSize=64
)

checkpoint = ModelCheckpoint('model{epoch:02d}.h5')

history = model.fit_generator(
    trainGenerator, 
    validation_data=validGenerator, 
    nb_val_samples=2560, 
    samples_per_epoch=23040, 
    nb_epoch=5, 
    verbose=2, 
    callbacks=[checkpoint]
)

print(model.summary())

# Save model data
model.save_weights('./model.h5')
json_string = model.to_json()
with open('./model.json', 'w') as f:
    f.write(json_string)