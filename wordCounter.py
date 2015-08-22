#lhy
#2015.7

import re
import cPickle
import MySQLdb
import numpy as np
from multiprocessing import Pool

def getTF(text,words,index):
    vectors = {}
    for word in words:
        vectors[word] = [0] * len(text)
    for i in range(len(text)):
        if i % (len(text) / 100) == 0:
            print "index " + str(index) + " processing, i = " + str(i)
        for word in words:
            vectors[word][i] = text[i].count(word)
    cPickle.dump(vectors,open("data/wordscount/" + str(index),'wb'))

class WordCounter():
    def __init__(self,words,text):
        self.words = words
        self.text = text

    def analysis1(self):
        vectors = {}
        for word in self.words:
            vectors[word] = [0] * len(self.text)
            for i in range(len(self.text)):
                vectors[word][i] = self.text[i].count(word)
        return vectors

    def analysis(self,proc_number):
        num = len(self.text)
        group = int(num / proc_number)
        start = 0
        p = Pool()
        for i in range(proc_number):
            if i != proc_number - 1:
                p.apply_async(getTF,args = (self.text[start:start + group],self.words,i))
                start += group
            else:
                p.apply_async(getTF,args = (self.text[start:num],self.words,i))
        p.close()
        p.join()
        vectors = {}
        for word in self.words:
            vectors[word] = []
        for i in range(proc_number):
            proc_vector = cPickle.load(open("data/wordscount/" + str(i),'rb'))
            for word in proc_vector.keys():
                vectors[word].extend(proc_vector[word])
        return vectors

    def getCount(self):
        inHandle = open(r'../text/20051105_pages_articles.hgw.xml')
        text = inHandle.read()
        vectors = {}
        wordDict = {}
        conn = MySQLdb.connect(host = "localhost",user = "root",passwd = "helloworld",db = "wiki",charset = "utf8",use_unicode = True)
        cursor = conn.cursor()
        cursor.execute("select term from idx;")
        data = cursor.fetchall()
        for i in range(len(data)):
            wordDict[data[i][0]] = 0
        #for word in self.words:
        #    vectors[word] = [0] * len(self.text)
        reToken = re.compile('[a-zA-Z]+')
        #for i in range(len(self.text)):
        for w in reToken.finditer(text):
            word = w.group()
            if word not in wordDict:
                continue
            wordDict[word] += 1
        wordList = sorted(wordDict.iteritems(),key = lambda x:x[1],reverse = True)
        wordList = wordList[0:2000]
        words = set([index[0] for index in wordList])
        cPickle.dump(words,open("data/words",'wb'))
        #return vectors

    def tfidf(self,proc_number):
        #vectors = self.analysis(20)
        vectors = self.getCount()
        for word in vectors.keys():
            df = 0
            for i in range(len(vectors[word])):
                if vectors[word][i] > 0:
                    df += 1
            for i in range(len(vectors[word])):
                if vectors[word][i] == 0:
                    continue
                value = float(vectors[word][i])
                vectors[word][i] = (1.0 + np.log(value)) * np.log(float(len(self.text)) / float(df))
        words = vectors.keys()
        for i in range(len(vectors[words[0]])):
            vector = []
            for word in words:
                if vectors[word][i] != 0:
                    vector.append(vectors[word][i])
            v = np.array(vector,dtype = float)
            vSum = np.sqrt((v ** 2).sum())
            for word in words:
                if vectors[word][i] != 0:
                    vectors[word][i] /= vSum
        cPickle.dump([len(self.text),vectors],open("data/tfidf",'wb'))
        print "TFIDF ok"

