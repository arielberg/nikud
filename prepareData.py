#!/usr/bin/python3

# This file should render a csv file for all the training data
# Each row in the csv should be a single training sample (a single char with nikod)
#
# Each sample should have also the scoring of the word it appears on and following
# and previous word using Word2Vec
import re
import hashlib
import csv
from collections import deque
import os
import fasttext

charBeforeAndAfter = 4

###################################
# Step 0: Prepare Word2Vec
###################################
def trainWordToVec():
    '''

    Will train word2vec from wikipedia text
    make sure that hewiki-latest-pages-articles.xml.bz2 is downloaded to
    the data folder

    this function creates word2vec model
    :return:

    '''

    print("WordToVec model exists: {}".format(os.path.isfile(word2VecFiles+".bin")))
    from gensim.corpora import WikiCorpus

    # stop if model already has been created
    if os.path.isfile(word2VecFiles+".bin"):
        return

    # download from wikipedia
    if not os.path.isfile(wikiTar):
        import urllib
        tarLocation = 'https://dumps.wikimedia.org/hewiki/latest/hewiki-latest-pages-articles.xml.bz2'
        urllib.request.urlretrieve(url, wikiTar)

    # parse tar file
    if not os.path.isfile(wikiFull):
        i = 0

        output = open(wikiFull, 'w')
        wiki = WikiCorpus(wikiTar, lemmatize=False, dictionary={})
        for text in wiki.get_texts():
            article = " ".join([t for t in text])
            output.write("{}\n".format(article))
            i += 1
            if (i % 500 == 0):
                print( "{} items loaded".format(str(i)))

        output.close()
    # train word2vec
    fasttext.skipgram(wikiFull, word2VecFiles)
    print("step 1 - {0} created".format(word2VecFiles))

###################################
# Step 1: Clean files
###################################
def cleanRowFiles():
    '''
        manipulate all text in sub directories to a single clean file
    :return:
    '''
    paths = ["/www/nikod/raw/bible", "/www/nikod/raw/stories"]

    if os.path.isfile(allTextsSingleFile):
        print('All text are parsed into single file')
        return
    with open(allTextsSingleFile, "w") as allTexts:
        for path in paths:
            for r, d, f in os.walk(path):
                for file in f:
                    if '.txt' in file:
                        with open(path + "/" + file, "r") as f:
                            lines = f.readlines();
                            lines = [re.sub("[^\." + chars + nikudStr + "]", "",line) for line in lines]
                            lines = [line for line in lines if next(nikod in line for nikod in nikudList)]
                            content = ".".join(lines)
                            content = re.sub(' +', ' ', content)
                            content = re.sub('\.+', '.', content)
                            lines = content.split(".")
                            lines = [line.strip() for line in lines]
                            lines = [line for line in lines if line != '']
                            content = "\n".join(lines)
                            allTexts.writelines(content)

            '''
           # np.savetxt("data/foo.csv", X, delimiter=",")
            # df = pd.DataFrame(X)
            # csvFile.write(X)
            # df.to_csv("data/foo.csv")
            #exit()
            '''
###################################
# Step 3: Get vectors for each char and save it to file
###################################
def createCharVector():
    with open(charVectorFile, "w") as charVectorWriter:
        csvCharVectorWriter = csv.writer(charVectorWriter)
        with open(allTextsSingleFile, "r") as allTexts:
            while True:
                line = allTexts.readline()
                if not line:
                    break
                X = sentenceToLabeledData(line)
                csvCharVectorWriter.writerows(X)


###################################
# Step 4: cleanUp
###################################
# create labels array (each label is a set of nikuds that should follow latters
def getNikodList():
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
    return nikodList,nikuds, nikudStr

def getWordVector(word, word2vec):
    return word2vec[word]

def sentenceToLabeledData(sentence):
    '''
    Prepare data for training

    @INPUT - content of full text

    @OUTPUT - csv of rows with the following structure:
        [1] char before
        [2] char after
        [3] current char
        [4] current vowel
        [5] words Vector
        [6] Boolean - has previous word
        [7] Boolean - has following word
        [8] char location in word
    '''
    print(sentence)
    # make sure only one space in the begining of a sentence
    sentence = sentence.strip(" ")
    words = deque( re.sub("[^" + chars + "]", "", sentence).split(" ") )
    word2vec = fasttext.load_model('model/wiki.he.fasttext.model.bin')

    # previousWordVector = getWordVector(words.popleft())

    sentenceWordsWindow = ["","",""]
    sentenceWordsWindow[1] = words.popleft()
    if len(words)>0:
        sentenceWordsWindow[2] = words.popleft()
    currentWordLen = len(sentenceWordsWindow[1]) -1
    charLocationInWord = -1
    wordsVector = getWordVector(" ".join(filter(None, sentenceWordsWindow)), word2vec)

    rows = [["", "", "", "", wordsVector, sentenceWordsWindow[0] == "", sentenceWordsWindow[2] == "", 0]]

    for charIndex in range(len(sentence)):
        c = sentence[charIndex]
        if c==" ":
            # update word vectors
            sentenceWordsWindow[0] = sentenceWordsWindow[1]
            sentenceWordsWindow[1] = sentenceWordsWindow[2]
            sentenceWordsWindow[2] = ""
            if len(words)>0:
                sentenceWordsWindow[2] = words.popleft()
                currentWordLen = len(sentenceWordsWindow[1]) - 1
                wordsVector = getWordVector(" ".join(filter(None, sentenceWordsWindow)), word2vec)

            charLocationInWord = -1

        hash = hashlib.md5(c.encode()).hexdigest()
        if c in chars:
            charLocationInWord += 1
            rows[len(rows)-1][1] = c

            for charIndex in range(1, charBeforeAndAfter):
                if len(rows)-charIndex > 0:
                    rows[len(rows)-charIndex-1][2] = c + rows[len(rows)-charIndex-1][2]
                    rows[len(rows)-1][3] += rows[len(rows)-charIndex-1][1]

            if currentWordLen == 0:
                charIndexInWord = 0
            else:
                charIndexInWord = (charLocationInWord / currentWordLen)
            rows.append(["", "", "", "", wordsVector, sentenceWordsWindow[0]=="", sentenceWordsWindow[2]=="",charIndexInWord])
        else:
            if hash=="68b329da9893e34099c7d8ad5cb9c940":
                continue
            if hash=="3389dae361af79b04c9c8e7057f60cc6":
                print(sentence)
                print(c)
                print(ord(c))
                continue
            rows[len(rows)-1][0]+=c


    return rows

chars = ' אבגדהוזחטיכלמנסעפצקרשתךםןףץ'
nikud,nikudList, nikudStr = getNikodList()
allTextsSingleFile = 'all_texts.txt'
charVectorFile = 'charVector.csv'
wikiTar = "data/hewiki-latest-pages-articles.xml.bz2"
wikiFull = "data/wiki.he.text"
word2VecFiles = "model/he.fasttext.model"
word2vec = None

if __name__== "__main__":
  # Step 0: Train word2vec
  trainWordToVec()

  # Step 1: Clean raw files
  cleanRowFiles()

  # Step 2: Create char vectore
  createCharVector()