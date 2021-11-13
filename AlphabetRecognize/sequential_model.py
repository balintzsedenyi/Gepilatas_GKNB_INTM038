from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data=pd.read_csv("A_Z\Dataset.csv").astype('float32')
X = data.drop('0',axis = 1)
y = data['0']
train_alph, test_alph, train_label, test_label = train_test_split(X, y, test_size = 0.2)

train_alph = np.reshape(train_alph.values, (train_alph.shape[0], 28,28))
test_alph = np.reshape(test_alph.values, (test_alph.shape[0], 28,28))

train_alph = train_alph.reshape(train_alph.shape[0],train_alph.shape[1],train_alph.shape[2],1)

test_alph = test_alph.reshape(test_alph.shape[0], test_alph.shape[1], test_alph.shape[2],1)

train_labelOHE = to_categorical(train_label, num_classes = 26, dtype='int')

test_labelOHE = to_categorical(test_label, num_classes = 26, dtype='int')

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Flatten())
model.add(Dense(64,activation ="relu"))
model.add(Dense(128,activation ="relu"))
model.add(Dense(26,activation ="softmax"))

model.compile(optimizer = Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_alph, train_labelOHE, epochs=1,  validation_data = (test_alph,test_labelOHE))
model.summary()
model.save(r'model.h5')
print("The validation accuracy is :", history.history['val_accuracy'])
print("The training accuracy is :", history.history['accuracy'])
print("The validation loss is :", history.history['val_loss'])
print("The training loss is :", history.history['loss'])