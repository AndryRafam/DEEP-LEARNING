""" Siamese network - CNN on MNIST """

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
from keras.layers import Dense, Dropout, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras.layers.merge import concatenate
from keras.datasets import mnist

# load data
(x_train,y_train), (x_test,y_test) = mnist.load_data()

# compute the number of labels
num_labels = len(np.unique(y_train))

# reshape and normalize input images
image_size = x_train.shape[1]
x_train = np.reshape(x_train,[-1,image_size,image_size,1])
x_test = np.reshape(x_test,[-1,image_size,image_size,1])
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

# network parameters
input_shape = (image_size,image_size,1)
batch_size = 32
kernel_size = 3
dropout = 0.4
n_filters = 32

# build model
def build_model():

	# left branch of Y network
	left_inputs = Input(shape=input_shape)
	x = left_inputs
	filters = n_filters

	# 3 layers of Conv2D-Dropout-MaxPooling2D
	# number of filters doubles after each layer (32-64-128)
	for i in range(3):
		x = Conv2D(filters=filters,
		           kernel_size=kernel_size, padding='same',
		           activation='relu')(x)
		x = Dropout(dropout)(x)
		x = MaxPooling2D()(x)
		filters *= 2
    
	# right branch of Y network
	right_inputs = Input(shape=input_shape)
	y = right_inputs
	filters = n_filters

	# 3 layers of Conv2D-Dropout-MaxPooling2D
	# number of filters doubles after each layer (32-64-128)
	for i in range(3):
		y = Conv2D(filters=filters, kernel_size=kernel_size,
		           padding='same', activation='relu',
		           dilation_rate=2)(y)
		y = Dropout(dropout)(y)
		y = MaxPooling2D()(y)
		filters *= 2

	# merge left and right branches outputs
	y = concatenate([x,y])

	# feature maps to vector before connecting to dense layer
	y = Flatten()(y)
	y = Dropout(dropout)(y)
	outputs = Dense(num_labels,activation='softmax')(y)

	# build the model in functional API
	return Model([left_inputs,right_inputs],outputs)

# verify the model
model = build_model()
model.summary()

# compile and train  the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adamax',
              metrics=['accuracy'])

model.fit([x_train,x_train],
          y_train, validation_data=([x_test,x_test],y_test),
          epochs=20,
          batch_size=batch_size)

# model accuracy on test dataset
score = model.evaluate([x_test,x_test],y_test,batch_size=batch_size)
print("\nTEST ACCURACY: %.1f%%" % (100*score[1]))
