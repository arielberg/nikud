import tensorflow as tf
import numpy as np
import hashlib
from tensorflow import feature_column
import pandas as pd
import utils
import fasttext

np.set_printoptions(linewidth=460)

tf.logging.set_verbosity(tf.logging.INFO)

def _model_fn(features, labels, mode, config):

  """Call the defined shared dnn_model_fn."""
  return _dnn_model_fn(
          features=features,
          labels=labels,
          mode=mode,
          head=head,
          hidden_units=hidden_units,
          feature_columns=tuple(feature_columns or []),
          optimizer=optimizer,
          activation_fn=activation_fn,
          dropout=dropout,
          input_layer_partitioner=input_layer_partitioner,
          config=config,
          batch_norm=batch_norm)




def do_train():
    word2vec = fasttext.load_model('model/wiki.he.fasttext.model.bin')

    DS = pd.read_csv(utils.charVectorFile, header=None, names=CSV_COLUMNS_TRAIN)
    '''
    DS = tf.data.experimental.make_csv_dataset(utils.charVectorFile, 100,
                                                header=False,
                                                select_columns= ['labels','word_is_beginig', 'word_is_ending','char_place_in_word'],
                                                label_name='labels',
                                                na_value="?",
                                                column_names=CSV_COLUMNS_TRAIN)
                                                '''
    DS['labels'] = DS['labels'].apply(lambda chars: chars if type(chars) is str else '')
   # DS['label_hotlist'] = DS['labels'].apply(lambda chars: [(1 if utils.nikudList[idx] in list(chars) else 0) for idx in range(len(utils.nikudList))])
    for idx in range(len(utils.nikudList)):
        DS['label_'+str(idx)] = DS['labels'].apply(lambda chars: 1 if utils.nikudList[idx] in list(chars) else 0)


    DS['wordVec'] = DS['window'].apply(lambda s: word2vec[s])
    DS = DS.join(pd.DataFrame(DS['wordVec'].tolist(), columns=["c" + str(i) for i in range(100)]))
    DS.pop('wordVec')

    for clm in CSV_COLUMNS_TRAIN:
        if clm not in ['word_is_beginig', 'word_is_ending', 'char_place_in_word']:
            DS.pop(clm);

    labelDS = DS[['label_'+str(idx) for idx in range(len(utils.nikudList))]]

    inputFnc = tf.estimator.inputs.pandas_input_fn(x=DS, y=labelDS['label_0'], shuffle=True)

    features_cols = [feature_column.numeric_column('word_is_beginig'),
                         feature_column.numeric_column('word_is_ending'),
                         feature_column.numeric_column('char_place_in_word')]
    features_cols.extend([feature_column.numeric_column("c" + str(i)) for i in range(100)])
    print(features_cols)
    estimator = nikudEstimator(
        feature_columns= features_cols,
        n_classes=len(utils.nikudList),
        model_dir= 'model/tmp',
        model_fn=model_fn,
        hidden_units=[1024, 512, 256])

    epochs = 10
    for epochId in range(0, epochs):
        batchId = 0


        batchId +=1
        print("**********************************")
        print("Epoch: "+str(epochId)+" Batch: "+str(batchId))
        print("**********************************")
        estimator.train(input_fn=inputFnc)

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

