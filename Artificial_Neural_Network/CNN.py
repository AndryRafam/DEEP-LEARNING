""" CNN (Convolutional Neural Network) on MNIST with BatchNormalization - 99,3% accuracy over 20 epochs """


### import libraries
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from keras.layers import Dense, Dropout, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import BatchNormalization, Activation
from keras.models import Model
from keras.datasets import mnist
from random import seed, randint
import numpy as np

### load data
(x_train,y_train), (x_test,y_test) = mnist.load_data()


### compute the number of labels
num_labels = len(np.unique(y_train))


### reshape and normalize input images
image_size = x_train.shape[1]
x_train = np.reshape(x_train,[-1,image_size,image_size,1])
x_test = np.reshape(x_test,[-1,image_size,image_size,1])
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255


### network parameters
input_shape = (image_size,image_size,1)
batch_size = 128
kernel_size = 3
filters = 64
dropout = 0.3


### functional API to build CNN layers
inputs = Input(shape=input_shape)
def build_model():
	y = inputs
	for i in range(3):
		y = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(y)
		y = BatchNormalization()(y)
		y = Activation('relu')(y)
		y = MaxPooling2D()(y)
	y = Flatten()(y)
	y = Dropout(dropout)(y)
	outputs = Dense(num_labels,activation='softmax')(y)
	return Model(inputs,outputs)

### build the model by supplying inputs/outputs
model = build_model()
model.summary()

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


### train the model
model.fit(x_train,y_train, validation_data=(x_test,y_test), epochs=20,batch_size=batch_size)


### model accuracy
score = model.evaluate(x_test,y_test,batch_size=batch_size)
print("\nScore: %.1f%%" % (100*score[1]))
