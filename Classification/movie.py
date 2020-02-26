#Classifying movie reviews: Binary Classification
#Using mse as loss function and tanh as activation function

from keras.datasets import imdb
from keras.models import Model
from keras.layers import Dense, Input
from keras import optimizers
import matplotlib.pyplot as plt
import numpy as np

(train_data,train_labels), (test_data,test_labels) = imdb.load_data(num_words=10000)

## Encoding the integer sequences into a binary matrix
def vectorize_sequences(sequences,dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i, sequence in enumerate(sequences):
        results[i,sequence]=1
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

## Model definition

epochs = 9
batch_size = 512
hidden_units = 16
input_shape = (10000,)
num_labels = len(np.unique(y_train))

def build_model():
	inputs = Input(shape=input_shape)
	y = Dense(hidden_units,activation='tanh')(inputs)
	y = Dense(hidden_units,activation='tanh')(y)
	outputs = Dense(num_labels,activation='sigmoid')(y)
	return Model(inputs,outputs)
	
model = build_model()
model.summary()

## Setting aside a validation set
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

## Compiling and training the model
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='mse',metrics=['acc'])

history = model.fit(partial_x_train,partial_y_train,epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(x_val,y_val))



history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc = history_dict['acc']

epochs = range(1,len(acc)+1)

## Plotting the training and validation loss
plt.plot(epochs,loss_values,'bo',label='Training loss')
plt.plot(epochs,val_loss_values,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

## Plotting the training and validation accuracy
plt.clf()
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

score = model.evaluate(x_test,y_test)
print("\nScore :%.1f%%" % (100.0*score[1]))
