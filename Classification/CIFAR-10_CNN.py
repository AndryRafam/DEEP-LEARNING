# CNN (Convolutional Neural Network) on CIFAR-10 - 77.3% accuracy

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Flatten, Dense, Activation
from keras.layers import Dropout, Input
from keras.datasets import cifar10
from keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt

# load cifar10 dataset
(x_train,y_train), (x_test,y_test) = cifar10.load_data()

# compute the number of labels
num_classes = len(np.unique(y_train))

# input image dimensions
image_size = x_train.shape[1]

# resize and normalize
x_train = np.reshape(x_train, [-3,image_size, image_size,3])
x_test = np.reshape(x_test, [-3,image_size,image_size,3])
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

# building model
batch_size = 32
input_shape = (image_size,image_size,3)
kernel_size = 3
n_filters = 32
epochs = 100

inputs = Input(shape=input_shape)

def build_model():
	y = inputs
	filters=n_filters
	for i in range(2):
		for j in range(2):
			y = Conv2D(filters, kernel_size, padding='same')(y)
			y = Activation('relu')(y)
		y = MaxPooling2D()(y)
		y = Dropout(0.25)(y)
		filters *= 2
	y = Flatten()(y)
	y = Dense(512)(y)
	y = Activation('relu')(y)
	y = Dropout(0.5)(y)
	outputs = Dense(num_classes, activation='softmax')(y)
	return Model(inputs,outputs)
	
model = build_model()
		
# build and train the network
opt = RMSprop(learning_rate=0.0001, decay=1e-6)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=epochs, batch_size=batch_size)
print("\n")

# print the test accuracy
score = model.evaluate(x_test,y_test)
print("\n\n\tTest accuracy: %.2f%%" % (100.0*score[1]))
print("\n\n")
