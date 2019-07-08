import tensorflow as tf
import numpy as np
import hashlib
from tensorflow import feature_column
import pandas as pd
import utils
import fasttext
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

np.set_printoptions(linewidth=460)

tf.logging.set_verbosity(tf.logging.INFO)

def get_model(use_word2vec, model_tmp_file = ''):
    # create model
    model = Sequential()

    inputSize = 246
    if use_word2vec:
        inputSize += 100

    # add model layers
    model.add(Dense(256, activation='relu', input_shape=(inputSize,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(len(utils.vowels_category.keys())))
    if model_tmp_file:
        model.load_weights(model_tmp_file)
    # compile model using mse as a measure of model performance
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def do_train():
    " Train Model "
    # Load Word to Vector model
    if (useWordVector):
        word2vec = fasttext.load_model('model/wiki.he.fasttext.model.bin')

    model = get_model(useWordVector)
    features_cols = [feature_column.numeric_column('word_is_beginig'),
                         feature_column.numeric_column('word_is_ending'),
                         feature_column.numeric_column('char_place_in_word')]
    features_cols.extend([feature_column.numeric_column("c" + str(i)) for i in range(100)])

    #train model
    batchSize = 1500
    epochs = 10

    for epochId in range(0, epochs):
        print("*****************************************")
        print("****        Epoch   "+str(epochId)+"      ********")
        print("*****************************************")
        DS = pd.read_csv(utils.charVectorFile, chunksize=80000, header=None, names=CSV_COLUMNS_TRAIN)

        for chunk in DS:
            # FIX nulls
            chunk['labels'] = chunk['labels'].apply(lambda chars: chars if type(chars) is str else '')
            chunk['prefix'] = chunk['prefix'].apply(lambda chars: chars if type(chars) is str else '')
            chunk['suffix'] = chunk['suffix'].apply(lambda chars: chars if type(chars) is str else '')

            # shuffle
            # chunk = chunk.sample(frac=1)

            chrIndex = 0;
            for c in utils.chars:
                if c == " ":
                    continue
                chunk['current_char_is_'+c] = chunk['char'].apply(lambda chars: 1 if chars == c else 0)

            for c in utils.chars:
                if c == " ":
                    continue
                for i in range(4):
                    chunk['before_'+str(i+1)+'_is_' + c] = chunk['prefix'].apply(lambda pref: 1 if len(pref)>i and pref[i] == c else 0)
                    chunk['after_'+str(i+1)+'_is_' + c] = chunk['suffix'].apply(lambda suff: 1 if len(suff)>i and suff[i] == c else 0)

            # DS['label_hotlist'] = DS['labels'].apply(lambda chars: [(1 if utils.nikudList[idx] in list(chars) else 0) for idx in range(len(utils.nikudList))])
            if (useWordVector):
                chunk['wordVec'] = chunk['window'].apply(lambda s: word2vec[s])
                wordVec = pd.DataFrame(chunk['wordVec'].tolist(), index= chunk.index, columns=["c" + str(i) for i in range(100)])

                chunk = chunk.join(wordVec)

                chunk.pop('wordVec')

            chunk['label_vowel'] = chunk['labels'].apply(
                lambda chars: [utils.vowels_category_map[ord(char)] for char in chars if ord(char) in utils.vowels_category_map])

            for vowel in utils.vowels_category.keys():

                chunk['label_'+ vowel] = chunk['label_vowel'].apply(
                    lambda v: 1 if vowel in v else 0)


            chunk.pop('label_vowel')

            for clm in CSV_COLUMNS_TRAIN:
                if clm not in ['word_is_beginig', 'word_is_ending', 'char_place_in_word']:
                    chunk.pop(clm);

            fields = list(chunk.dtypes.keys())
            lastColX = fields[fields.index('label_a')-1]
            #Y = chunk[['label_' + str(idx) for idx in range(len(utils.nikudList))]]
            X = np.asarray(chunk.loc[:, :lastColX])
            Y = np.asarray(chunk.loc[:, 'label_a':])


            model.fit(X, Y) #, nb_epoch=5)
            model_type = 'without_word2vec';
            if(useWordVector):
                model_type = 'with_word2vec';
            model.save_weights("model/vowels/vowels_"+model_type+"_256_128_62_32_"+str(epochId)+".h5")

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
                        'suffix',
                        'prefix',
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
useWordVector = True

if __name__== "__main__":
    do_train()

