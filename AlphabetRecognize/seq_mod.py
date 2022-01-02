
"""
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import keras
from keras import layers
"""

from keras.layers.core.dropout import Dropout
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Rescaling

IMG_WIDTH=24
IMG_HEIGHT=24
DATA_DIR=r"D:\School things\DigitalVision\trainset"
NUM_CLASSES = 62

trainset=tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="grayscale",
    batch_size=64,
    image_size=(IMG_WIDTH, IMG_HEIGHT),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="training",
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)
valset=tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="grayscale",
    batch_size=64,
    image_size=(IMG_WIDTH, IMG_HEIGHT),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="validation",
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

"""
class_names = trainset.class_names
print(class_names)

label=os.listdir(DATA_DIR)
filelist=[]
img_db=0
label_s=[]
cl=0.0
for i in label:
      folder=DATA_DIR+"\\"+i
      for file in os.listdir(folder):
            img_db+=1
            label_s.append(cl)
            filelist.append(os.path.join(folder,file))
            if img_db%1900==0:
                  break
            else:
                  continue
                
      cl+=1.0
            
#img_dset = np.array([cv2.imread(fname,0) for fname in filelist ])
for fname in filelist:
      img_dset=np.array(cv2.imread(fname,0).astype('float32'))
img_dset=img_dset/255.0

train_ds, test_ds, train_l, test_l=train_test_split(dataset, labelset, test_size=0.2)

train_ds = train_ds.reshape(train_ds.shape[0],train_ds.shape[1],train_ds.shape[2],1)
test_ds = test_ds.reshape(test_ds.shape[0], test_ds.shape[1], test_ds.shape[2],1)

train_lb = to_categorical(train_l, num_classes = NUM_CLASSES, dtype='int')
test_lb = to_categorical(test_l, num_classes = NUM_CLASSES, dtype='int')

model = keras.Sequential(
    [
        keras.Input(shape=(IMG_HEIGHT,IMG_WIDTH,1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
)"""
model = Sequential()
model.add(Rescaling(1./255))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(IMG_HEIGHT,IMG_WIDTH,1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(64,activation ="relu"))
model.add(Dense(128,activation ="relu"))
model.add(Dense(NUM_CLASSES,activation ="softmax"))
#model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"],)
model.compile(optimizer = Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(trainset, batch_size=64, epochs=1,  validation_data = valset)
model.summary()
score=model.evaluate(valset,verbose=0)
print(score)
model.save(r'98numbonly.h5')
print("The validation accuracy is :", history.history['val_accuracy'])
print("The training accuracy is :", history.history['accuracy'])
print("The validation loss is :", history.history['val_loss'])
print("The training loss is :", history.history['loss'])
