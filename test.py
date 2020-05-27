from keras.utils import *
from keras import losses
import numpy as np
import sys
import cv2
import os
from sklearn.metrics import *
from keras.models import load_model

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
    size =1
    frames = 1

    path = sys.argv[1]
    name = getListOfFiles(path)
    
    name = getListOfFiles(sys.argv[1])
    #images might not be in order, need to sort them
    name.sort()
    x=[]
    y=[]
    prediction = []
    model = load_model('model.h5')
    for i, fileN in enumerate(name):
        if i%size ==0 and i != 0:
            x = np.array(x)
            x = x[:, :, :, :, np.newaxis]
            #print(x.shape)
            #labels = to_categorical(y)
            #print(labels)
            prediction.append(model.predict(x))
            #print(str(len(prediction)) + " length")
            x=[]
        #process frames in a video
        frameList = []
        video = cv2.VideoCapture(fileN)
        print(i)
        while True:
            done, frame = video.read()
            if int(video.get(cv2.CAP_PROP_POS_FRAMES)) % frames != 0:
                continue
            if not done:
                video.release()
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (int(1280/8), int(720/8)))
            frameList.append(frame)
        x.append(frameList)
        tempName = fileN.split("/")
        #print(tempName)
        if tempName[-2] == "positive":
            y.append(1)
        else:
            y.append(0)
    right = 0
    maxi = []
    outfile = open('out.txt', 'w+')
    for i in prediction:
        outfile.write(str(i) + ', ')
    outfile.close()
    for value in prediction:
        if value[0] > .5:
            maxi.append(1)
        else:
            maxi.append(0)
    for i, value in enumerate(maxi): 
        #print(str(y[i]) + " " + str(value))
        if int(y[i]) == value:
            right = right + 1
    print(right/len(y))
