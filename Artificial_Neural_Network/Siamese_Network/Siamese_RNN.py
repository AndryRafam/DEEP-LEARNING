""" Siamese network - RNN on MNIST """

### import libraries
import numpy as np
from keras.layers import Dense, Input, SimpleRNN
from keras.models import Model
from keras.datasets import mnist
from keras.layers.merge import concatenate
import matplotlib.pyplot as plt

### load data
(x_train,y_train), (x_test,y_test) = mnist.load_data()

### compute the number of labels
num_labels = len(np.unique(y_train))

### reshape and normalize input images
image_size = x_train.shape[1]
x_train = np.reshape(x_train,[-1,image_size,image_size])
x_test = np.reshape(x_test,[-1,image_size,image_size])
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

### visualize the data
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

### network parameters
input_shape = (image_size,image_size)
units = 256
dropout = 0.4
epochs = 20

### left branch of the network
left_inputs = Input(shape=input_shape)
x = SimpleRNN(units=units,dropout=dropout)(left_inputs)

### right branch of the network
right_inputs = Input(shape=input_shape)
y = SimpleRNN(units=units,dropout=dropout)(right_inputs)

### merge left and right branches
z = concatenate([x,y])

### outputs
outputs = Dense(num_labels,activation='softmax')(z)

### build the model
model = Model([left_inputs,right_inputs],outputs=outputs)
model.summary()

### compile and train the model
model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',
	         metrics=['accuracy'])
model.fit([x_train,x_train],y_train,validation_data=([x_test,x_test],y_test),
          epochs=20)


### evaluate the model
score = model.evaluate([x_test,x_test],y_test)
print("\nTest accuracy: %.1f%%" % (100*score[1]))
