#encoding:utf-8

#lhy
#2015.7

import re
import cPickle
import numpy as np
import xml.etree.cElementTree as cElementTree
import lxml.html as html
import xmlwikiprep
import wordCounter

class ESA():

    def __init__(self,route = "/data/disk1/private/cxx/WikiNoDisamWord2VecTrain.txt"):
        self.text = []

    def getTags(self):
        filename = "/data/disk1/private/cxx/WordEmbeddingWithTags/notnull/enwiki2014_cate_notnull.txt"
        fileTags = []
        inHandle = open(filename,'r')
        lines = inHandle.readlines()
        for i in range(len(lines)):
            line = lines[i].lower().strip('\n').split(' ')
            fileTags.append(line)
        return fileTags

    def getWordSim(self):
        inHandle = open(r'../../wordsim353/combined.csv')
        wordsimList = inHandle.readlines()
        words = set([])
        for i in range(1,len(wordsimList)):
            line = wordsimList[i].lower().strip('\n').split(',')
            for j in range(2):
                if line[j] not in words:
                    words.add(line[j])
        return words

    def getSelected(self):
        #Get selected wikipedia articles
        inHandle1 = open(r'/home/lhy/ESA/wikiprep-esa/selected.txt')
        lines = inHandle1.readlines()
        inHandle2 = open(r'/home/lhy/ESA/text/20051105_pages_articles.hgw.xml')
        selectedID = set([])
        selectedArticles = []
        self.linkNum = []
        self.idList = []
        self.inLinkDict = {}
        for i in range(len(lines)):
            selectedID.add(int(lines[i].strip('\n')))
        for _id in selectedID:
            self.inLinkDict[_id] = 0
        for doc in xmlwikiprep.read(inHandle2):
            page_id = int(doc["_id"])
            for link in doc["links"]:
                linkID = int(link)
                if linkID in selectedID:
                    self.inLinkDict[linkID] += 1
            if page_id not in selectedID:
                continue
            #for link in doc["links"]:
            #    linkID = int(link)
            #    if linkID in selectedID:
            #        self.inLinkDict[linkID] += 1.0
            self.idList.append(page_id)
            title = doc["title"]
            title = html.fromstring(title).text_content().lower()
            text = doc["text"]
            text = html.fromstring(text).text_content().lower()
            mergeList = [title]
            mergeList.append(text)
            mergeText = ' '.join(mergeList)
            selectedArticles.append(mergeText)
            #self.linkNum.append(len(doc["links"]))
        self.text = selectedArticles
        words = self.getWordSim()
        #counter = wordCounter.WordCounter(words,self.text)
        #counter.tfidf(20)
        print "SelectedArticles ok"

    def parseConcept(self):
        words = self.getWordSim()
        fileTags = self.getTags()
        concepts = {}
        index = 0
        for i in range(len(fileTags)):
            for j in range(len(fileTags[i])):
                if fileTags[i][j] not in concepts:
                    concepts[fileTags[i][j]] = index
                    index += 1
        vectors = {}
        wordConcepts = {}
        conceptsNum = len(concepts.keys())
        for word in words:
            vectors[word] = [0] * conceptsNum
            wordConcepts[word] = set([])
        for i in range(len(self.text)):
            line = self.text[i].strip('\n').split(' ')
            for j in range(len(line)):
                if line[j] not in words:
                    continue
                else:
                    for concept in fileTags[i]:
                        wordConcepts[line[j]].add(concept)
                        vectors[line[j]][concepts[concept]] += 1
        for (key,value) in vectors.items():
            for i in range(len(value)):
                value[i] = float(value[i]) * float(conceptsNum) / float(len(wordConcepts[key]))
        cPickle.dump([conceptsNum,vectors],open("data/vectors",'wb'))

    def parseArticle(self):
        self.getSelected()
        words = self.getWordSim()
        self.row = []
        self.column = []
        self.data = []
        in_file = {}
        columnNum = 0
        textList = []
        #reToken = re.compile("[^ \t\n\r`~!@#$%^&*()_=+|\[;\]\{\},./?<>:’'\\\\\"]+")
        reToken = re.compile('[a-zA-Z]+')
        reAlpha = re.compile("^[a-zA-Z\-_]+$")
        #Index pruning
        vectors = cPickle.load(open("data/tfidf",'rb'))
        for (key,value) in vectors[1].items():
            wordVector = {}
            for i in range(len(value)):
                if value[i] == 0:
                    continue
                #value[i] /= vSum
                #value[i] *= np.log(1 + np.log(1.0 + self.inLinkDict[self.idList[i]]))
                wordVector[i] = value[i] * np.log(1 + np.log(1.0 + self.inLinkDict[self.idList[i]]))
                #value[index] *= np.log(1 + np.log(1.0 + self.inLinkDict[self.idList[index]]))
            v = sorted(wordVector.iteritems(),key = lambda x : x[1],reverse = True)
            windowSize = 500
            #v.extend([(-1,0)] * windowSize)
            max_value = v[0][1]
            remainSet = set([])
            truncateSet = set([])
            index = 0
            mark = 0
            windowMark = 0
            first = 0
            last = 0
            window = [0] * windowSize
            for i in range(len(v)):
                window[windowMark] = v[i][1]
                if mark == 0:
                    first = v[i][1]
                    last = v[i][1]
                if mark < windowSize:
                    remainSet.add(v[i][0])
                elif first - last > max_value * 0.05:
                    remainSet.add(v[i][0])
                    if windowMark < windowSize - 1:
                        first = window[windowMark + 1]
                    else:
                        first = window[0]
                else:
                    break
                last = v[i][1]
                mark += 1
                windowMark += 1
                windowMark = windowMark % windowSize
            for i in range(len(vectors[1][key])):
                if i not in remainSet:
                    vectors[1][key][i] = 0
                else:
                    vectors[1][key][i] = wordVector[i]
        cPickle.dump(vectors,open("data/vectors",'wb'))
        print "Vectors ok"
