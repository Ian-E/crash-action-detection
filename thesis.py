import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, TimeDistributed, Conv3D, LSTM, Flatten
from keras.layers import *
from keras.utils import *
from keras import losses
import numpy as np
import sys
import cv2
import os
import random
from sklearn.model_selection import train_test_split

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
    return allFiles

if __name__ == "__main__":
    size =1282
    frames = 2
    model = Sequential()
    #model.add(ConvLSTM2D(filters=16, kernel_size=(3, 5)
                      #, data_format='channels_last'
                      #, recurrent_activation='hard_sigmoid'
                      #, activation='tanh'
                      #, padding='same', return_sequences=True, input_shape=(int(100/frames),int(720/8),int(1280/8),3)))
    model.add(Conv3D(16, kernel_size=(3, 6, 13), kernel_regularizer=keras.regularizers.l2(.01), data_format="channels_last", input_shape=(int(100/frames),int(720/8),int(1280/8), 3)))
    #input_shape=(int(100/frames),int(720/8),int(1280/8), 3)
    #model.add(Conv3D(32, kernel_size=(5,9,16), data_format="channels_last"))
    model.add(LeakyReLU())
    model.add(Dropout(rate=.5))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 3, 3)))
    model.add(Conv3D(32, kernel_size=(3,6, 13), data_format="channels_last", kernel_regularizer=keras.regularizers.l2(.01)))
    model.add(LeakyReLU())
    model.add(Dropout(rate=.3))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    #model.add(Conv3D(128, kernel_size=(3,3,3), data_format="channels_last", activation='relu'))
    #model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    #model.add(Conv3D(256, kernel_size=(1,1,1), data_format="channels_last", activation='relu'))
    #model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    #normalize data before sending to lstm
   
    model.add(ConvLSTM2D(filters=64, kernel_size=(9, 16)
                       , data_format='channels_last'
                       , recurrent_activation='hard_sigmoid'
                       , activation='tanh'
                       , padding='same', return_sequences=True))
    model.add(Dropout(rate=.1))
    #flatten to send to dense layers
    model.add(Flatten())
    #model.add(Dense(512))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(keras.optimizers.SGD(learning_rate=0.006),loss='binary_crossentropy',metrics=['accuracy'])
    
    x=np.load('x.npy')

    x_val = np.load('x_val.npy')

    y = np.load('y.npy')

    y_val = np.load('y_val.npy')

    #callbacks = [
        #keras.callbacks.EarlyStopping(
            ## Stop training when `val_loss` is no longer improving
            #monitor='val_loss',
            ## "no longer improving" being defined as "no better than 1e-2 less"
            #min_delta=0,
            ## "no longer improving" being further defined as "for at least 2 epochs"
            #patience=3,
            #verbose=1)
    #]
    #print(labels)
    model.fit(x, y, epochs=5, batch_size=5, validation_data=(x_val, y_val))


   

    model.save('model.h5')
    
