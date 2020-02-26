### import libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Input
from keras.models import Model
from keras.datasets import mnist
import matplotlib.pyplot as plt

### load data
(x_train,y_train), (x_test,y_test) = mnist.load_data()

### computer the number of labels
num_labels = len(np.unique(y_train))

### sample and plot 25 mnist digits from train dataset
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


### image dimensions (assumed square)
image_size = x_train.shape[1]
input_shape = image_size*image_size

### resize and normalize
x_train = np.reshape(x_train, [-1, input_shape])
x_train = x_train.astype('float32')/255
x_test = np.reshape(x_test, [-1, input_shape])
x_test = x_test.astype('float32')/255

### network parameters
batch_size = 128
hidden_units = 256
dropout = 0.45


### 3-layer MLP model with relu and dropout after each layer

inputs = Input(shape=(input_shape,))
for i in range(2):
	y = Dense(hidden_units)(inputs)
	y = Activation('relu')(y)
	y = Dropout(dropout)(y)

y = Dense(num_labels)(y)
outputs = Activation('softmax')(y)

model = Model(inputs,outputs)

model.summary()

### compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train,y_train, epochs=20, batch_size=batch_size)
print("\n")
print(history.history.keys())

### plotting
plt.plot(history.history['accuracy'],'bo', label='accuracy')
plt.plot(history.history['loss'], 'ro', label='loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([-0.5,1.5])
plt.legend(loc='best')
plt.show()

### validate the model on test dataset to determmine generalization
score = model.evaluate(x_test, y_test, batch_size=batch_size)
print("\nTest accuracy: %.1f%%" % (100.0*score[1]))
print("\n")
