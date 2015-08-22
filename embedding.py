#lhy
#2015.8

import ESA
import cPickle
import MySQLdb
import struct
import numpy as np
import word2vec
import nimfa
from scipy import sparse
from multiprocessing import Pool

wordDict = {}
articleDict = {}
matrix = ""
W = ""
S = ""
H1 = ""
H2 = ""

def softmax(i):
    return 1.0 / (1.0 + np.exp(-1.0 * i))

def gradient(L,p,i):
    return (L - p) * i

def getItem(l,index):
    try:
        return l[index]
    except:
        print "Index Error"

def loadVectors(words,proc_id):
    row = []
    column = []
    matrix_data = []
    conn = MySQLdb.connect(host = "localhost",user = "root",passwd = "helloworld",db = "wiki",charset = "utf8",use_unicode = True)
    cursor = conn.cursor()
    command = "select vector from idx where term = '%s'"
    row = []
    column = []
    matrix_data = []
    wordNum = 0
    print "Load Vectors"
    for word in words:
        cursor.execute(command % word)
        s = cursor.fetchall()[0][0]
        s_len = struct.unpack(">i",s[0:4])[0]
        s = s[4:]
        for i in range(s_len):
            index = struct.unpack(">if",s[0:8])
            row.append(wordDict[word])
            column.append(articleDict[int(index[0])])
            matrix_data.append(float(index[1]))
            s = s[8:]
    cPickle.dump([row,column,matrix_data],open("data/proc/" + str(proc_id),'wb'))
    print "Process " + str(proc_id) + " ok"

def optimize(a,b):
    global matrix
    global W
    global S
    global H1
    global H2
    for i in range(a,b):
        rate = 1
        v = list(matrix.getrow(i).toarray())[0]
        v = np.array(map((lambda x:(x > 0)),v),dtype = float)
        for j in range(max_iter):
            e = np.transpose(H2).dot(S[i])
            f = np.array(map(softmax,e))
            print H2.shape
            for k in range(H2.shape[0]):
                H2[k] += rate * alpha * np.array(map(gradient,v,f,[1] * H2.shape[1])) * S[i][k]
            for k in range(len(S[i])):
                S[i][k] += rate * alpha * np.array(map(gradient,v,f,H2[k])).sum()
                if S[i][k] < 0:
                    S[i][k] = 0
            r = S[i]
            S[i] -= rate * alpha * np.transpose(H1).dot(H1.dot(S[i]) - W[i]) + 1e-8
            S[i] = np.array(map((lambda x:(x > 0)),S[i]),dtype = float)
            for k in range(H1.shape[0]):
                H1[k] -= rate * alpha * (H1[k].dot(r) - W[i][k]) * r + 2e-8 * H1[k]
            rate *= 0.98

class MC():

    def __init__(self,route):
        self.model = word2vec.load(route)
        self.words = list(self.model.vocab)

    def loadMatrix(self,proc_number):
        global wordDict
        global articleDict
        global matrix
        inHandle = open(r'../wikiprep-esa/selected.txt')
        in_list = inHandle.readlines()
        selected = []
        for i in range(len(in_list)):
            selected.append(in_list[i].strip('\n'))
        for i in range(len(selected)):
            articleDict[int(selected[i])] = i
        words = cPickle.load(open("data/words",'rb'))
        self.wordList = []
        for word in self.words:
            if word in words:
                self.wordList.append(word)
        cPickle.dump(self.wordList,open("data/wordList",'wb'))
        for i in range(len(self.wordList)):
            wordDict[self.wordList[i]] = i
        rowNum = len(wordDict.keys())
        columnNum = len(articleDict.keys())
        #group = int(len(self.wordList) / proc_number)
        #start = 0
        #p = Pool()
        #for i in range(proc_number):
        #    if i != proc_number - 1:
        #        p.apply_async(loadVectors,args = (self.wordList[start:start + group],i))
        #        start += group
        #    else:
        #        p.apply_async(loadVectors,args = (self.wordList[start:],i))
        #p.close()
        #p.join()
        #while loaded_proc != proc_number:
        #    pass
        row = []
        column = []
        matrix_data = []
        for i in range(proc_number):
            data = cPickle.load(open("data/proc/" + str(i)))
            row.extend(data[0])
            column.extend(data[1])
            matrix_data.extend(data[2])
        print "Vectors ok"
        row = np.array(row,dtype = int)
        column = np.array(column,dtype = int)
        matrix_data = np.array(matrix_data,dtype = np.float32)
        matrix = np.transpose(sparse.csr_matrix((matrix_data,(row,column)),shape = (rowNum,columnNum)))

    def mf(self,k,max_iter,alpha):
        global matrix
        global W
        vectors = [list(self.model[word]) for word in self.wordList]
        W = np.array(vectors,dtype = np.float32)
        del vectors
        H = np.transpose(np.random.rand(W.shape[1],len(articleDict)))
        for i in range(H.shape[0]):
            rate = 1
            v = self.matrix.getrow(i).toarray()
            for j in range(max_iter):
                g = W.dot(H[i]) - v
                H[i] -= rate * alpha * np.transpose(W).dot(np.array(g[0]))
                H[i] = np.array(map((lambda x:max(x,0)),H[i]))
                rate *= 0.5
        print "Begin NMF"
        nmf = nimfa.Nmf(sparse.csr_matrix(np.transpose(H)),seed = "random_vcol",rank = k,max_iter = 10)
        nmf_fit = nmf()
        H1 = nmf_fit.basis()
        H2 = nmf_fit.coef()
        print "NMF ok"
        construct = W.dot(H1.toarray())
        wordVectors = {}
        myESA = ESA.ESA()
        wordsim = myESA.getWordSim()
        for i in range(len(construct)):
            if self.wordList[i] in wordsim:
                wordVectors[self.wordList[i]] = construct[i]
        cPickle.dump([columnNum,wordVectors],open("data/mc_matrix",'wb'))
        print "MC ok"

    def nn(self,k,max_iter,alpha,proc_number):
        global matrix
        global W
        global S
        global H1
        global H2
        matrix = np.transpose(matrix)
        vectors = [list(self.model[word]) for word in self.wordList]
        W = np.array(vectors,dtype = np.float32)
        del vectors
        H1 = np.random.rand(W.shape[1],k) / 10000
        S = np.random.rand(W.shape[0],k) / 10000
        H2 = np.random.rand(k,len(articleDict)) / W.shape[1]
        group = int(S.shape[0] / proc_number)
        start = 0
        p = Pool()
        for i in range(proc_number):
            if i != proc_number - 1:
                p.apply_async(optimize,args = (start,start + group))
                start += group
            else:
                p.apply_async(optimize,args = (start,S.shape[0]))
        p.close()
        p.join()
        wordVectors = {}
        myESA = ESA.ESA()
        wordsim = myESA.getWordSim()
        for i in range(len(S)):
            if self.wordList[i] in wordsim:
                wordVectors[self.wordList[i]] = list(S[i])
        cPickle.dump([columnNum,wordVectors],open("data/mc_matrix",'wb'))

if __name__ == "__main__":
    myMC = MC("text8.txt")
    myMC.loadMatrix(20)
    myMC.nn(1000,200,0.05,20)
    print "WordVectors ok"
