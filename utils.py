import hashlib

import sys


def getNikodList():
    '''
    Reads all valid nikud from text file and convert it to
    hash based dictionary


    :return: list of nikuds, hash base dictionary, string of all valid nikud
    '''

    nikods = {hashlib.md5("".encode()).hexdigest():""}
    nikudStr = ""
    f = open("nikud.txt", "r")

    line = f.readline()
    while line:
        line = f.readline()
        line = line.strip()
        linep = line.split(':')
        if(len(linep)==2):
            nikods[linep[0]] = linep[1]
            nikudStr = nikudStr+linep[1]
    f.close()

    nikodList = list(nikods)
    nikuds = list(nikods.values())
    nikuds.remove('')
    return nikodList, nikuds, nikudStr

# How many chars should be vectorized before/atter the trained char
charBefore = 4
charAfter = 4

# all valid chars & space
chars = ' אבגדהוזחטיכלמנסעפצקרשתךםןףץ'

allTextsSingleFile = 'all_texts.txt'
charVectorFile = 'charVector.csv'

wikiTar = "/hewiki-latest-pages-articles.xml.bz2"
wikiFull = "/wiki.he.text"

word2VecFiles = "/he.fasttext.model"


nikud,nikudList, nikudStr = getNikodList()

data_location = 'data/'
data_location = next((arg.replace('--DATA_PATH=','') for arg in sys.argv if 'DATA_PATH' in arg),data_location)

wikiTar = data_location+"hewiki-latest-pages-articles.xml.bz2"
wikiFull = data_location+"/wiki.he.text"

model_locaion = 'model/'
model_locaion = next((arg.replace('--MODEL_PATH=','') for arg in sys.argv if 'MODEL_PATH' in arg),model_locaion)

word2VecFiles = model_locaion+"/he.fasttext.model"

vector_CSV_file = 'play.csv' #'charsVectorsWithWordsVector.csv'
print(word2VecFiles)
