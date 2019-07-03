import tensorflow as tf
import numpy as np
import hashlib
from tensorflow import feature_column
import pandas as pd
import utils
import fasttext
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping

np.set_printoptions(linewidth=460)

tf.logging.set_verbosity(tf.logging.INFO)


def do_train():
    " Train Model "

    # Load Word to Vector model
    word2vec = fasttext.load_model('model/wiki.he.fasttext.model.bin')

    features_cols = [feature_column.numeric_column('word_is_beginig'),
                         feature_column.numeric_column('word_is_ending'),
                         feature_column.numeric_column('char_place_in_word')]
    features_cols.extend([feature_column.numeric_column("c" + str(i)) for i in range(100)])

    #create model
    model = Sequential()

    #add model layers
    model.add(Dense(150, activation='relu', input_shape=(355,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(len(utils.nikudList)))

    #compile model using mse as a measure of model performance
    model.compile(optimizer='adam', loss='mean_squared_error')

    #train model
    batchSize = 1500
    epochs = 10

    for epochId in range(0, epochs):
        print("*****************************************")
        print("****        Epoch   "+str(epochId)+"      ********")
        print("*****************************************")
        DS = pd.read_csv(utils.charVectorFile, chunksize=100000, header=None, names=CSV_COLUMNS_TRAIN)

        for chunk in DS:
            chunk['labels'] = chunk['labels'].apply(lambda chars: chars if type(chars) is str else '')
            chunk['prefix'] = chunk['prefix'].apply(lambda chars: chars if type(chars) is str else '')
            chunk['suffix'] = chunk['suffix'].apply(lambda chars: chars if type(chars) is str else '')

            # shuffle
            chunk = chunk.sample(frac=1)

            chrIndex = 0;
            for c in utils.chars:
                chunk['current_char_is_'+str(chrIndex)] = chunk['labels'].apply(lambda chars: 1 if type(chars) is c else 0)
                for i in range(4):
                    chunk['current_before_'+str(i)+'_is_' + str(chrIndex)] = chunk['prefix'].apply(lambda pref: 1 if len(pref)>i and pref[i] is c else 0)
                    chunk['current_after_'+str(i)+'_is_' + str(chrIndex)] = chunk['suffix'].apply(lambda suff: 1 if len(suff)>i and suff[i] is c else 0)
                chrIndex=chrIndex+1

            # DS['label_hotlist'] = DS['labels'].apply(lambda chars: [(1 if utils.nikudList[idx] in list(chars) else 0) for idx in range(len(utils.nikudList))])

            chunk['wordVec'] = chunk['window'].apply(lambda s: word2vec[s])
            wordVec = pd.DataFrame(chunk['wordVec'].tolist(), index= chunk.index, columns=["c" + str(i) for i in range(100)])

            chunk = chunk.join(wordVec)

            chunk.pop('wordVec')

            for idx in range(len(utils.nikudList)):
                chunk['label_' + str(idx)] = chunk['labels'].apply(
                    lambda chars: 1 if utils.nikudList[idx] in list(chars) else 0)

            for clm in CSV_COLUMNS_TRAIN:
                if clm not in ['word_is_beginig', 'word_is_ending', 'char_place_in_word']:
                    chunk.pop(clm);

            #Y = chunk[['label_' + str(idx) for idx in range(len(utils.nikudList))]]
            X = np.asarray(chunk.loc[:, :'c99'])
            Y = np.asarray(chunk.loc[:, 'label_0':])

           #  X = tf.convert_to_tensor(X)
           # Y = np.reshape(Y, (len(Y), len(utils.nikudList)))

            model.fit(X, Y)
            model.save_weights("model/tmp/model_100X50_B_"+str(epochId)+".h5")

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

CSV_COLUMNS_TRAIN = [   'labels',
                        'char',
                        'prefix',
                        'suffix',
                        'window',
                        'word_is_beginig',
                        'word_is_ending',
                        'char_place_in_word',
                     ]
'''
for i in range(100):
    CSV_COLUMNS_TRAIN.append("word_to_vec_"+str(i))
'''
chunkID = -1
chunkedDS = None

if __name__== "__main__":
    do_train()

