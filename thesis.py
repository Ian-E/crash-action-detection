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
    #model.add(ConvLSTM2D(filters=16, kernel_size=(3, 3)
                      #, data_format='channels_last'
                      #, recurrent_activation='hard_sigmoid'
                      #, activation='tanh'
                      #, padding='same', return_sequences=True, input_shape=(int(100/frames),int(720/8),int(1280/8),3)))
    model.add(Conv3D(32, kernel_size=(3, 32, 18), input_shape=(int(100/frames),int(720/8),int(1280/8), 3)))
    #input_shape=(int(100/frames),int(720/8),int(1280/8), 3)
    model.add(ReLU())
    model.add(TimeDistributed(BatchNormalization()))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    model.add(Conv3D(64, kernel_size=(3,32,18), data_format="channels_last"))
    model.add(LeakyReLU())
    model.add(TimeDistributed(BatchNormalization()))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    #model.add(Conv3D(128, kernel_size=(3,3,3), data_format="channels_last", activation='relu'))
    #model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    #model.add(Conv3D(256, kernel_size=(1,1,1), data_format="channels_last", activation='relu'))
    #model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    #normalize data before sending to lstm
   
    model.add(ConvLSTM2D(filters=64, kernel_size=(5, 3)
                       , data_format='channels_last'
                       , recurrent_activation='hard_sigmoid'
                       , activation='tanh'
                       , padding='same', return_sequences=True))
    model.add(TimeDistributed(Conv2D(128, kernel_size=(3,3), data_format="channels_last")))
    model.add(ReLU())
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    #flatten to send to dense layers
    model.add(Flatten())
    #model.add(Dense(512))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(keras.optimizers.Adadelta(learning_rate=0.01),loss='binary_crossentropy',metrics=['accuracy'])
    path = sys.argv[1]
    name = getListOfFiles(path)
    
    name = getListOfFiles(sys.argv[1])
    #Shuffle images so positive and negative are next ot each other
    random.shuffle(name)
    #name.sort();
    x=[]
    y=[]
    for i, fileN in enumerate(name):
        #if i%size ==0 and i != 0:
        #    x=np.array(x)
        #    print(x.shape)
        #    labels = to_categorical(y)
            #print(labels)
        #    model.fit(x, y, epochs=300, batch_size=15)
        #    x=[]
        #    y=[]
        #process frames in a video
        frameList = []
        video = cv2.VideoCapture(fileN)
        while True:
            done, frame = video.read()
            if int(video.get(cv2.CAP_PROP_POS_FRAMES)) % frames != 0:
                continue
            if not done:
                video.release()
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (int(1280/8), int(720/8)))
            frameList.append(frame)
        x.append(frameList)
        video = cv2.VideoCapture(fileN)
        frameList = []
        
        tempName = fileN.split("/")
        print(tempName)
        if tempName[-2] == "positive":
            y.append(1)
            y.append(1)
            while True:
                done, frame = video.read()
                if int(video.get(cv2.CAP_PROP_POS_FRAMES)) % frames != 0:
                    continue
                if not done:
                    video.release()
                    break
                frame = cv2.flip(frame, 1)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (int(1280/8), int(720/8)))
            
                frameList.append(frame)
            x.append(frameList)
        else:
            y.append(0)
    #print(x.shape)
    x, x_val, y, y_val = train_test_split(x, y, test_size=.1)
    x=np.array(x)
    x_val = np.array(x_val)
    y = np.array(y)
    y_val = np.array(y_val)
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
    model.fit(x, y, epochs=7, batch_size=5, validation_data=(x_val, y_val))


   

    model.save('model.h5')
    
