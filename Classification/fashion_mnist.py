"""Fashion Mnist with CNN (Convolutional Neural Network)"""

# import libraries
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
from keras.models import Model
from keras.layers import Conv2D, Flatten, MaxPooling2D
from keras.layers import Dropout, Dense, Input, BatchNormalization
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from random import randint, seed

# loading the data
(x_train,y_train), (x_test,y_test) = fashion_mnist.load_data()

# verify that the data are correct
indexes = np.random.randint(0,x_train.shape[0], size=25)
images = x_train[indexes]
labels = y_train[indexes]

plt.figure(figsize=(5,5))
for i in range(len(indexes)):
    plt.subplot(5,5,i+1)
    image = images[i]
    plt.imshow(image,cmap='gray')
    plt.axis('off')

plt.show()
plt.close('all')

# reshape and normalize
num_labels = len(np.unique(y_train))
image_size = x_train.shape[1]
x_train = np.reshape(x_train,[-1,image_size,image_size,1])
x_test = np.reshape(x_test,[-1,image_size,image_size,1])
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

# build model
input_shape = (image_size,image_size,1)
batch_size = 32
dropout = 0.25
filters = 32
epochs = 20
kernel_size = 5

def build_model(filters,kernel_size,dropout):
    inputs = Input(shape=input_shape)
    for i in range(3):
        cnn = Conv2D(filters=filters,kernel_size=kernel_size,padding='same',
                   activation='relu')(inputs)
        cnn = BatchNormalization()(cnn)
        cnn = Dropout(dropout)(cnn)
        cnn = MaxPooling2D()(cnn)
        filters *= 2
    cnn = Flatten()(cnn)
    cnn = Dropout(dropout)(cnn)
    outputs = Dense(num_labels,activation='softmax')(cnn)
    return Model(inputs,outputs)

if __name__ == '__main__':
	model = build_model(filters,kernel_size,dropout)

	# train model
	model.compile(optimizer='adamax',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=epochs,batch_size=batch_size)

	# accuracy
	score = model.evaluate(x_test,y_test,batch_size=batch_size)
	print("\nACCURACY: %.1f%%" % (100*score[1]))

	# Let's make some prediction
	class_number = [0,1,2,3,4,5,6,7,8,9]
	for i in range(5):
		seed()
		image_index = randint(0,10000)
		plt.imshow(x_test[image_index].reshape(image_size,image_size),cmap='gray')
		plt.xlabel(class_number[y_test[image_index]])
		plt.show()
		pred = model.predict(x_test[image_index].reshape(-1,image_size,image_size,1))
		print("PREDICTION = ", pred.argmax(), " Image Class = ", class_number[y_test[image_index]])
		print("\n")
