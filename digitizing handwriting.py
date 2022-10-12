# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 10:50:58 2022

@author: xiw
"""

# Title: Image cropping and digitize handwriting
# Contact: wxi@leegov.com
# Author: Wang Xi
# Last Updated: 06-24-2022

# Update notes: convert scanned pdf into png 
# File Format: png

# Import libraries
import os
import pytesseract
from pytesseract import Output
import re
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import keras
from keras.datasets import mnist # import data
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
import tensorflow as tf
import seaborn as sns
from sklearn.model_selection import train_test_split
from itertools import groupby

# Find tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Set up working directory
os.chdir(r'C:\Users\xiw\Desktop')
os.getcwd()

# Load image
tram = cv2.imread(r'C:\Users\xiw\Desktop\image.png')
plt.imshow(tram)
plt.show()

# Load image1
tram1 = cv2.imread(r'C:\Users\xiw\Desktop\image1.png')
plt.imshow(tram1)
plt.show()

# Cut ridership image
cut_totalr = tram[250: 350, 400: 600] # img[y:y+h, x:x+w]
cv2.imshow("Cut Image of Total Ridership", cut_totalr)
cv2.waitKey()

testr = Image.fromarray(cut_totalr)
testr = testr.save("testr.png")

# Cut ridership image1
cut_totalr1 = tram1[200: 300, 400: 600] # img[y:y+h, x:x+w]
cv2.imshow("Cut Image of Total Ridership", cut_totalr1)
cv2.waitKey()

testr1 = Image.fromarray(cut_totalr1)
testr1 = testr1.save("testr1.png")

# Cut ridership image1-alternative
cut_totalr2 = tram1[620: 700, 0: 150] # img[y:y+h, x:x+w]
cv2.imshow("Cut Image of Total Ridership", cut_totalr2)
cv2.waitKey()

testr2 = Image.fromarray(cut_totalr2)
testr2 = testr2.save("testr2.png")

# Detect only digits
custom_config = r'--oem 3 --psm 6 outputbase digits'
total_number = pytesseract.image_to_string(cut_totalr, config=custom_config)
total_number

# Detect only digits
custom_config = r'--oem 3 --psm 6 outputbase digits'
total_number1 = pytesseract.image_to_string(cut_totalr1, config=custom_config)
total_number1

# Detect only digits-alternative
custom_config = r'--oem 3 --psm 6 outputbase digits'
total_number2 = pytesseract.image_to_string(cut_totalr2, config=custom_config)
total_number2


# Cut date box
cut_date = tram[50: 100, 430: 600]
cv2.imshow("Cut Date Box", cut_date)
cv2.waitKey()

# Adding custom options
date = pytesseract.image_to_string(cut_date, config = '--psm 12')
date






# Testing Approach 1
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train.shape
plt.imshow(x_train[0])
plt.show()
plt.imshow(x_train[0], cmap = plt.cm.binary)

# we don't need care about the color change after normalization
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

plt.imshow(x_train[0], cmap = plt.cm.binary)

IMG_SIZE = 28
# increasing one dimension for kernel operation
x_trainr = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
x_testr = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
print("Training Sample dimension", x_trainr.shape)
print("Testing Sample dimension", x_testr.shape)

# Creating a neural network now
model = Sequential()

# First convolution layer (60000, 28, 28, 1)  28-3+1 = 26x26
model.add(Conv2D(64, (3,3), input_shape = x_trainr.shape[1:])) # only for first convolution layer to mention input layer size
model.add(Activation("relu")) # activation function to make it non-linear, if 0 remove; if >0 keep
model.add(MaxPooling2D(pool_size=(2,2))) # maxpooling single maximum value of 2x2

# 2nd convolution layer # pooling size is 2, 26/2=13, 13-3+1 = 11x11
model.add(Conv2D(64, (3,3))) # 2nd convolution layer
model.add(Activation("relu")) # activation function
model.add(MaxPooling2D(pool_size=(2,2))) # maxpooling

# 3rd convolution layer # pooling size is 2, 11/2=5, 5-3+1 = 3x3
model.add(Conv2D(64, (3,3))) # 3rd convolution layer
model.add(Activation("relu")) # activation function
model.add(MaxPooling2D(pool_size=(2,2))) # maxpooling

# Fully connected layer # 1
model.add(Flatten()) # before using fully connected layer, need to be flatten so that 2D to 1D
model.add(Dense(64))
model.add(Activation("relu"))

# Fully connected layer # 2
model.add(Dense(32))
model.add(Activation("relu"))

# Last fully connected layer, output must be equal to number of classes, 10 (0-9)
model.add(Dense(10)) # the last layer must be 10 because we are predicting the probabilities of 10 types of different numbers
model.add(Activation('softmax')) # activation function is changed to softmax (class probabilities)

model.summary()

print ("Total Training Samples = ", len(x_trainr))
x_trainr.shape

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(x_trainr, y_train, epochs = 5, validation_split = 0.3)
# validation's accuracy should be almost equal to accuracy
# if validation accuracy is way smaller than accuracy, we are suffering the overfitting, in this case, we should drop out the layers

# Evaluate on testing data set MNIST
test_loss, test_acc = model.evaluate(x_testr, y_test)
print("Test Loss on 10,000 test samples", test_loss)
print("Validation Accuracy on 10,000 test samples", test_acc)



# import test png
# test = cv2.imread("test3.png")
# plt.imshow(test)
# plt.show()

# rows, cols, _ = test.shape
# print("Rows", rows)
# print("Columns", cols)
# test.shape

# copy and paste
# Loading image in grayscale
image = Image.open("test7.png").convert("L")


# resizing to 28 height pixels
w = image.size[0]
h = image.size[1]
r = w / h # aspect ratio
new_w = int(r * 28)
new_h = 28
new_image = image.resize((new_w, new_h))

# converting to a numpy array
new_image_arr = np.array(new_image)

# inverting the image to make background = 0
new_inv_image_arr = 255 - new_image_arr

# rescaling the image
final_image_arr = new_inv_image_arr / 255.0
# plt.imshow(new_inv_image_arr)
# plt.imshow(final_image_arr)

# splitting image array into individual element arrays using non zero columns
m = final_image_arr.any(0)
out = [final_image_arr[:,[*g]] for k, g in groupby(np.arange(len(m)), lambda x: m[x] != 0) if k]


'''
iterating through the element arrays to resize them to match input 
criteria of the model = [mini_batch_size, height, width, channels]
'''

num_of_elements = len(out)
elements_list = []
for x in range(0, num_of_elements):
    img = out[x]
    
    #adding 0 value columns as fillers
    width = img.shape[1]
    filler = (final_image_arr.shape[0] - width) / 2
    
    if filler.is_integer() == False:    #odd number of filler columns
        filler_l = int(filler)
        filler_r = int(filler) + 1
    else:                               #even number of filler columns
        filler_l = int(filler)
        filler_r = int(filler)
    
    arr_l = np.zeros((final_image_arr.shape[0], filler_l)) #left fillers
    arr_r = np.zeros((final_image_arr.shape[0], filler_r)) #right fillers
    
    #concatinating the left and right fillers
    help_ = np.concatenate((arr_l, img), axis= 1)
    element_arr = np.concatenate((help_, arr_r), axis= 1)
    
    element_arr.resize(28, 28, 1) #resize array 2d to 3d
    #storing all elements in a list
    elements_list.append(element_arr)


elements_array = np.array(elements_list)

# reshaping to fit model input criteria
elements_array = elements_array.reshape(-1, 28, 28, 1)

# plt.imshow(elements_array[0])
# plt.imshow(elements_array[1])
# plt.imshow(elements_array[2])
# plt.imshow(elements_array[3])

# predicting using the created model
elements_pred =  model.predict(elements_array)
elements_pred = np.argmax(elements_pred, axis = 1)
# elements_array.shape
# len(out)
print(elements_pred)




# test = test[:, :, 0]
test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

resized = cv2.resize(test, (28,28), interpolation = cv2.INTER_AREA)

resized.shape

newimg = tf.keras.utils.normalize (resized, axis =1) # 0 to 1 scaling

newimg = np.array(newimg).reshape(-1, IMG_SIZE, IMG_SIZE, 1) # kernel operation of convolution layer
newimg.shape

predictions = model.predict(newimg)
print(np.argmax(predictions))




