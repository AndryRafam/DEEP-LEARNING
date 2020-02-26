import numpy as np
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Flatten, Dense, Dropout, Input
from keras.datasets import cifar10
from random import randint
import matplotlib.pyplot as plt

# load cifar10 dataset
(x_train,y_train), (x_test,y_test) = cifar10.load_data()

# compute the number of labels
num_labels = len(np.unique(y_train))

# sample and plot cifar10 images from train dataset
indexes = np.random.randint(0,x_train.shape[0], size=25)
images = x_train[indexes]
labels = y_train[indexes]

plt.figure(figsize=(5,5))
for i in range(len(indexes)):
    plt.subplot(5,5,i+1)
    image = images[i]
    plt.imshow(image)
    plt.axis('off')

plt.show()
plt.close('all')

# input image dimensions
image_size = x_train.shape[1]

# resize and normalize
x_train = np.reshape(x_train, [-3,image_size, image_size,3])
x_test = np.reshape(x_test, [-3,image_size,image_size,3])
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

# parameters
input_shape = (image_size,image_size,3)
kernel_size = 3
n_filters = 32
dropout = 0.3
epochs = 5

# building model
inputs = Input(shape=input_shape)
def build_model():
	filters = n_filters
	for i in range(3):
		y = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu')(inputs)
		y = Dropout(dropout)(y)
		y = MaxPooling2D()(y)
		filters *= 2
		
	y = Flatten()(y)
	y = Dropout(dropout)(y)
	outputs = Dense(num_labels,activation='softmax')(y)
	return Model(inputs,outputs)

model = build_model()
model.summary()

# build and train the network
model.compile(optimizer='rmsprop',loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
history = model.fit(x_train,y_train,epochs=epochs)
print("\n")
print(history.history.keys())

# print the test accuracy
score = model.evaluate(x_test,y_test)
print("\n\n\tTest accuracy: %.2f%%" % (100.0*score[1]))
print("\n\n")

# plotting
plt.plot(history.history['accuracy'],'b',label='accuracy')
plt.plot(history.history['loss'],'r',label='loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0,2])
plt.legend(loc='upper right')
plt.show()

# make some prediction
image_index = randint(0,10000)
plt.imshow(x_test[image_index].reshape(image_size,image_size),cmap='gray')
plt.show()
pred = model.predict(x_test[image_index].reshape(-3,image_size,image_size,3))
print(pred.argmax())
