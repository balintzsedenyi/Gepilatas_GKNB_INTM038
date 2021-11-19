import os
import cv2
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D

IMG_WIDTH=128
IMG_HEIGHT=128
DATA_DIR=r"D:\School things\DigitalVision\trainset"
NUM_CLASSES = 62

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
            
img_dset = np.array([cv2.imread(fname,0) for fname in filelist ])

train_ds, test_ds, train_l, test_l=train_test_split(img_dset, label_s, test_size=0.2)

train_ds = train_ds.reshape(train_ds.shape[0],train_ds.shape[1],train_ds.shape[2],1)
test_ds = test_ds.reshape(test_ds.shape[0], test_ds.shape[1], test_ds.shape[2],1)

train_lb = to_categorical(train_l, num_classes = NUM_CLASSES, dtype='int')
test_lb = to_categorical(test_l, num_classes = NUM_CLASSES, dtype='int')

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(IMG_HEIGHT,IMG_WIDTH,1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Flatten())
model.add(Dense(64,activation ="relu"))
model.add(Dense(128,activation ="relu"))
model.add(Dense(NUM_CLASSES,activation ="softmax"))


model.compile(optimizer = Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_ds, train_lb, epochs=3,  validation_data = (test_ds,test_lb))
model.summary()
model.save(r'model_test.h5')
print("The validation accuracy is :", history.history['val_accuracy'])
print("The training accuracy is :", history.history['accuracy'])
print("The validation loss is :", history.history['val_loss'])
print("The training loss is :", history.history['loss'])