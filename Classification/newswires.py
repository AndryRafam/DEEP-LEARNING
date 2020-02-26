# Classifying newswires: Mutliclass classification

from keras.datasets import reuters
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.merge import concatenate
import matplotlib.pyplot as plt

(train_data,train_labels), (test_data,test_labels) = reuters.load_data(num_words=10000)


### Encoding the data
def vectorize_sequences(sequences,dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i, sequence in enumerate(sequences):
        results[i,sequence]=1
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.array(train_labels)
y_test = np.array(test_labels)


### network parameters
hidden_units = 64
epochs = 9
batch_size = 512
input_shape = (10000,)
num_labels = len(np.unique(y_train))

### Model definition
left_inputs = Input(shape=input_shape)
x = Dense(hidden_units,activation='relu')(left_inputs)
x = Dense(hidden_units,activation='relu')(x)

right_inputs = Input(shape=input_shape)
y = Dense(hidden_units,activation='relu')(right_inputs)
y = Dense(hidden_units,activation='relu')(y)

z = concatenate([x,y])

outputs = Dense(num_labels,activation=('softmax'))(z)

model = Model([left_inputs,right_inputs],outputs)

### Compiling the model
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


### Validating the approach
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = y_train[:1000]
partial_y_train = y_train[1000:]


### Training the model
history = model.fit([partial_x_train,partial_x_train],
                    partial_y_train,
                    batch_size=batch_size,
                    validation_data=([x_val,x_val],y_val),
                    epochs=epochs)



### Plotting the training and validation loss
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(loss)+1)

plt.plot(epochs,loss,label='Training loss')
plt.plot(epochs,val_loss,label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()



### Plotting the training and validation accuracy
plt.clf()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs,acc,label='Training acc')
plt.plot(epochs,val_acc,label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

score = model.evaluate([x_test,x_test],y_test)
print("\nScore: %.1f%%" % (100.0*score[1]))


