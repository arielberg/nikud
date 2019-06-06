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
import utils

charBeforeAndAfter = 4

###################################
# Step 0: Prepare Word2Vec
###################################
def trainWordToVec():
    """

    Will train word2vec from wikipedia text
    make sure that hewiki-latest-pages-articles.xml.bz2 is downloaded to
    the data folder

    this function creates word2vec model
    :return:

    """

    print("WordToVec model exists: {}".format(os.path.isfile(utils.word2VecFiles + ".bin")))
    from gensim.corpora import WikiCorpus

    # stop if model already has been created
    if os.path.isfile(utils.word2VecFiles + ".bin"):
        return

    # download from wikipedia
    if not os.path.isfile(utils.wikiTar):
        import urllib
        tarLocation = 'https://dumps.wikimedia.org/hewiki/latest/hewiki-latest-pages-articles.xml.bz2'
        wikiConn = urllib.request.urlopen(tarLocation)
        with open(utils.wikiTar, 'wb') as wikiSaver:
            wikiSaver.write(wikiConn.read())

    # parse tar file
    if not os.path.isfile(utils.wikiFull):
        i = 0

        output = open(utils.wikiFull, 'w')
        wiki = WikiCorpus(utils.wikiTar, lemmatize=False, dictionary={})
        for text in wiki.get_texts():
            article = " ".join([t for t in text])
            output.write("{}\n".format(article))
            i += 1
            if i % 500 == 0:
                print("{} items loaded".format(str(i)))

        output.close()
    # train word2vec
    fasttext.skipgram(utils.wikiFull, utils.word2VecFiles)
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
                            lines = [re.sub("[^\." + chars + nikudStr + "]", "", line) for line in lines]
                            lines = [line for line in lines if next(nikod in line for nikod in nikudList)]
                            content = ".".join(lines)
                            content = re.sub(' +', ' ', content)
                            content = re.sub('\.+', '.', content)
                            lines = content.split(".")
                            lines = [line.strip() for line in lines]
                            lines = [line for line in lines if line != '']
                            content = "\n".join(lines)
                            allTexts.writelines(content)


###################################
# Step 3: Get vectors for each char and save it to file
###################################
def createCharsCSV():
    with open(utils.charVectorFile, "w") as charVectorWriter:
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
    nikods = {hashlib.md5("".encode()).hexdigest(): ""}
    nikudStr = ""
    f = open("nikud.txt", "r")

    line = f.readline()
    while line:
        line = f.readline()
        line = line.strip()
        linep = line.split(':')
        if (len(linep) == 2):
            nikods[linep[0]] = linep[1]
            nikudStr = nikudStr + linep[1]
    f.close()

    nikodList = list(nikods)
    nikuds = list(nikods.values())
    nikuds.remove('')
    return nikodList, nikuds, nikudStr

def sentenceToLabeledData(sentence):
    '''
    Prepare data for training

    @INPUT - content of full text

    @OUTPUT - csv of rows with the following structure:
        [1] char before
        [2] char after
        [3] current char
        [4] current vowel
        [5] current sentence part (3 words)
        [6] current word
        [7] char location in word
    '''
    # make sure only one space in the begining of a sentence
    sentence = sentence.strip(" ")
    words: deque = deque(re.sub("[^" + chars + "]", "", sentence).split(" "))

    sentence_words_window = ["", "", ""]
    sentence_words_window[1] = words.popleft()
    if len(words) > 0:
        sentence_words_window[2] = words.popleft()
    current_word_len = len(sentence_words_window[1]) - 1
    charLocationInWord = -1
    wordsVector = " ".join(filter(None, sentence_words_window))

    rows = [["", "", "", "", wordsVector, sentence_words_window[0] == "", sentence_words_window[2] == "", 0]]

    for charIndex in range(len(sentence)):
        c = sentence[charIndex]
        if c == " ":
            # update word vectors
            sentence_words_window[0] = sentence_words_window[1]
            sentence_words_window[1] = sentence_words_window[2]
            sentence_words_window[2] = ""
            if len(words) > 0:
                sentence_words_window[2] = words.popleft()
                current_word_len = len(sentence_words_window[1]) - 1
                wordsVector = " ".join(filter(None, sentence_words_window))

            charLocationInWord = -1

        if c in chars:
            charLocationInWord += 1
            rows[len(rows) - 1][1] = c

            for charIndex in range(1, charBeforeAndAfter):
                if len(rows) - charIndex > 0:
                    rows[len(rows) - charIndex - 1][2] = c + rows[len(rows) - charIndex - 1][2]
                    rows[len(rows) - 1][3] += rows[len(rows) - charIndex - 1][1]

            if current_word_len == 0:
                charIndexInWord = 0
            else:
                charIndexInWord = (charLocationInWord / current_word_len)
            rows.append(["", "", "", "", wordsVector, sentence_words_window[0] == "", sentence_words_window[2] == "",
                         charIndexInWord])
        elif c in nikudStr:
            rows[len(rows) - 2][0] += c
        else:
                print(sentence)
                print(c)
                print(ord(c))
                continue


    return rows


def create_vectors_csv():
    word2vec = fasttext.load_model('model/wiki.he.fasttext.model.bin')
    charDfChunked = pd.read_csv('charVector.csv', chunksize=10000, header=None,
                                names=['label', 'char', 'pre', 'suf', 'window', 'isStart', 'isEnd', 'weight'])

    with open('charsVectorsWithWordsVector.csv', 'a') as csvFile:
        batchId = 1
        for df in charDfChunked:
            df['label'] = df['label'].apply(lambda s: s if type(s) == str else "")
            df['pre'] = df['pre'].apply(lambda s: s if type(s) == str else "")
            df['suf'] = df['suf'].apply(lambda s: s if type(s) == str else "")
            df['label'] = df['label'].astype(str)

            df['label_hotlist'] = df['label'].apply(
                lambda s: [(1 if utils.nikudStr[i] in s else 0) for i in range(len(utils.nikudStr))])

            df.reset_index(drop=True, inplace=True)

            df = df.join(pd.DataFrame(np.zeros(shape=(df.shape[0], len(utils.nikudStr))),
                                      columns=["id" + str(i) for i in range(len(utils.nikudStr))]))

            hotlist_length = utils.charBefore * len(utils.chars) + utils.charAfter * len(utils.chars) + 1
            df = df.join(pd.DataFrame(np.zeros(shape=(df.shape[0], hotlist_length)),
                                      columns=[
                                          "char_" + str(int(i / len(utils.chars))) + "_" + str(i % len(utils.chars)) for
                                          i in range(hotlist_length)]))
            for i in range(len(utils.chars)):
                df['char_0_' + str(i)] = df['char'].apply(lambda s: 1 if s == utils.chars[i] else 0)

            for i in range(utils.charBefore):
                for c in range(len(utils.chars)):
                    df['char_' + str(i + 1) + '_' + str(c)] = df['pre'].apply(
                        lambda s: 1 if len(s) > i and s[i] == utils.chars[c] else 0)

            for i in range(utils.charAfter):
                for c in range(len(utils.chars)):
                    df['char_' + str(i + 1 + utils.charBefore) + '_' + str(c)] = df['suf'].apply(
                        lambda s: 1 if len(s) > i and s[i] == utils.chars[c] else 0)

            df['wordVec'] = df['window'].apply(lambda s: word2vec[s])

            df.index = range(df.shape[0])
            df = df.join(pd.DataFrame(df['wordVec'].tolist(), columns=["c" + str(i) for i in range(100)]))

            del df['wordVec']
            del df['window']
            del df['char']
            del df['pre']
            del df['suf']
            del df['label']

            df.to_csv(csvFile, index=False, header=None)
            print('preparing vectors: bach {} is done'.format(batchId))
            batchId = batchId + 1


chars = ' אבגדהוזחטיכלמנסעפצקרשתךםןףץ'
nikud, nikudList, nikudStr = utils.getNikodList()
allTextsSingleFile = 'all_texts.txt'

word2vec = None

if __name__ == "__main__":
    # Step 0: Train word2vec
    trainWordToVec()

    # Step 1: Clean raw files
    cleanRowFiles()

    # Step 2: Create char vectore
    createCharsCSV()

    # Step 3: Create vectors
    create_vectors_csv()
