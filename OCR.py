import sys 
import cv2
import numpy as np
import matplotlib.pyplot as plt 
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,Flatten ,Dropout
from keras.models import load_model
from keras.layers import Convolution2D, MaxPooling2D
import keras
import os
from sklearn import preprocessing

img_gen=ImageDataGenerator(rescale=1./255, validation_split=0.2)
            
train_set=img_gen.flow_from_directory('C:/Users/AMIN/Desktop/PLT/DATA',
                                                          target_size=(60,120),
                                                          batch_size=50,
                                                          class_mode='categorical',
                                                          color_mode='grayscale',
                                                          subset='training',
                                                          seed=1)
            
valid_set=img_gen.flow_from_directory('C:/Users/AMIN/Desktop/PLT/DATA',
                                                          target_size=(60,120),
                                                          batch_size=50,
                                                          class_mode='categorical',
                                                          color_mode='grayscale',
                                                          subset='validation',
                                                          seed=1)




model = Sequential()
model.add(Convolution2D(32,4, activation='relu',input_shape=(60,120,1)))
model.add(Convolution2D(32, 4, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(900, activation='relu'))
model.add(Dense( 9, activation='softmax'))
                        
         
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001, name='Adam'),
              metrics=['accuracy'])
            
          
history = model.fit(x=train_set, validation_data=valid_set,
                                epochs=100, verbose=1)

model.save("model_jadid.h5")
print("مدل ذخیره شد")

x=max(history.history['accuracy'])
y=min(history.history['loss'])
xx=max(history.history['val_accuracy'])
yy=min(history.history['val_loss']) 

print("amozesh-deghat", x)
print("test-deghat", xx)





# min_max_scaler = preprocessing.MinMaxScaler()
# train_set = min_max_scaler.fit_transform(train_set)
# valid_set = min_max_scaler.fit_transform(valid_set)





