import tensorflow as tf
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping
import hashlib
import math
import random
import csv


np.set_printoptions(linewidth=460)

tf.logging.set_verbosity(tf.logging.INFO)

def rawDataToLabeledData(csvRow):
    '''
    The training should be called after prepareData has cleaned the string and set
    word vectors for the context of each char

    :param sentence: csv raw from prepareData.py result
    :return: feature vectors and label
    '''
    X = np.zeros((7, len(chars))) #
    Y = np.zeros((len(nikods)))

    for nikud in csvRow[0]:
        hash = hashlib.md5(nikud.encode()).hexdigest()
        Y[nikodList.index(hash)] = 1

    X[0][chars.index(csvRow[1])] = 1
    i = 1
    for columnIndex in [2,3]:
        for char in csvRow[columnIndex]:
            X[i][chars.index(char)] = 1
            i=i+1
    X= X.flatten()
    print(csvRow[4])
    X = np.concatenate([X, csvRow[4]])
    X = np.concatenate([X, [csvRow[5],csvRow[6],csvRow[7]]])
    print(X)
    return X, Y

def doTrain():

    lines = []
    with open(charVectorFile, "r") as charFileReader:
        csvCharReader = csv.reader(charFileReader)
        for line in csvCharReader:
            # eval is evil
            # TODO: Do NOT use eval - NEVER :)
            line[4] = eval(line[4])
            lines.append(line)

    #create model
    model = Sequential()

    inputS = len(chars)* 21
    #add model layers
    model.add(Dense(100, activation='relu', input_shape=(inputS,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(len(nikods)))

    #compile model using mse as a measure of model performance
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stopping_monitor = EarlyStopping(patience=3)

    #train model
    batchSize = 1500
    epochs = 10
    charsInSentence = 180
    for epochId in range(0, epochs):
        random.shuffle(lines)
        batchId = 0

        for batchLines in np.array_split(lines, (math.ceil(len(lines) / batchSize)+1)):
            batchId +=1
            print("**********************************")
            print("Epoch: "+str(epochId)+" Batch: "+str(batchId))
            print("**********************************")
            X = []
            Y = []
            for i in range(0, len(batchLines)):
                if batchLines[i][1] == "":
                    continue
                Xi,Yi = rawDataToLabeledData(batchLines[i])
                X.append(Xi)
                Y.append(Yi)

            print(len(X))
            for xi in X:
                print(len(xi))
            X = tf.convert_to_tensor(X)
            print(X.shape)
            Y = np.reshape(Y, (len(Y), len(nikods)))

            print(X.shape)
            print(Y.shape)
            model.fit(X, Y, validation_split=0.2, epochs=2, callbacks=[early_stopping_monitor])
        model.save_weights("weights/model_100X50_B_"+str(epochId)+".h5")

def getNikuds():
    '''
    Helper function to get all forms of nikud list and dictionaries

    TODO: Clean and move to Utility
    :return:
    '''
    # create labels array (each label is a set of nikuds that should follow latters
    nikods = {hashlib.md5("".encode()).hexdigest(): ""}
    f = open("nikud.txt", "r")

    line = f.readline()
    while line:
        line = f.readline()
        line = line.strip()
        linep = line.split(':')
        if (len(linep) == 2):
            nikods[linep[0]] = linep[1]
    f.close()

    nikodList = list(nikods)
    return nikodList, nikods

####  Start  ####
''' 
    TODO: Move to utility file
'''
chars = ' אבגדהוזחטיכלמנסעפצקרשתךםןףץ'
nikodList, nikods = getNikuds()
charVectorFile = 'charVectorTmp.csv'

if __name__== "__main__":
    doTrain()
