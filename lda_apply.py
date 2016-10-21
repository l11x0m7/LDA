# -*- encoding:utf-8 -*-
import sys
import os
reload(sys)
import numpy as np
sys.setdefaultencoding('utf-8')

class LDAApply():
    def __init__(self):
        self.word2id = dict()
        self.id2word = dict()
        self.id_cut_docs = list()
        self.z = list()
        self.theta = list() # doc->topic
        self.phi = list()   # topic->word
        self.loadParas()
        self.similarity()


    def loadParas(self):
        with open('data/result/word2id', 'rb') as fr:
            for line in fr:
                items = line.strip().split('\t')
                word = items[0].decode()
                wid = int(items[1])
                self.word2id[word] = wid
                self.id2word[wid] = word

        with open('data/result/id_cut_docs', 'rb') as fr:
            for line in fr:
                items = line.strip().split('\t')
                self.id_cut_docs.append(items)


        with open('data/result/z', 'rb') as fr:
            for line in fr:
                items = line.strip().split('\t')
                self.z.append(items)
            self.z = np.array(self.z)

        with open('data/result/theta', 'rb') as fr:
            for line in fr:
                items = line.strip().split('\t')
                self.theta.append(items)
            self.theta = np.array(self.theta)

        with open('data/result/phi', 'rb') as fr:
            for line in fr:
                items = line.strip().split('\t')
                self.phi.append(items)
            self.phi = np.array(self.phi)

