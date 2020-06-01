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
def getFrames(fileN, aug):
    framelist= []
    video = cv2.VideoCapture(fileN)
    while True:
        done, frame = video.read()
        if int(video.get(cv2.CAP_PROP_POS_FRAMES)) % frames != 0:
            continue
        if not done:
            video.release()
            break
        if aug:
            frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (int(1280/divisor), int(720/divisor)))
        frameList.append(frame)
    return frameList
    

if __name__ == "__main__":
    size =1282
    frames = 2
    divisor = 8
    aug = True
    
    name = getListOfFiles(sys.argv[1])
    #Shuffle images so positive and negative are next to each other
    random.shuffle(name)
    name.sort();
    x=[]
    y=[]
    for i, fileN in enumerate(name):
        print(i)
        #process frames in a video
        frameList = getFrames(fileN, False)
        x.append(frameList)
        #data augmentation
        if aug:
            frameList = getFrames(fileN, True)
            x.append(frameList)
        tempName = fileN.split("/")
        if tempName[-2] == "positive":
            y.append(1)
            if aug:
                y.append(1)
            
        else:
            y.append(0)
            if aug:
                y.append(0)
    #print(x.shape)
    x, x_val, y, y_val = train_test_split(x, y, test_size=.1)
    np.save('x', x)
    #if one channel use this format for x and x_val
    #x = x[:, :, :, :, np.newaxis]
    x_val = np.load('x_val.npy')
    np.save('x_val', x_val)
    np.save('y', y)
    np.save('y_val', y_val)
