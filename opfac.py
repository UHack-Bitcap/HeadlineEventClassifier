import csv
import gensim
import os
import collections
import smart_open
import random
import numpy as np
from sklearn.naive_bayes import GaussianNB
def read_corpus(fname):
    with open(fname, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='|', quotechar='|')
        for row in spamreader:
           # print row
	    if(len(row)==0):
		continue
            # For training data, add tags
	    text = str(row[0])
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(text), [1])
if __name__=='__main__':
    op_file = 'event.csv'
    train_corpus = list(read_corpus(op_file))
    train = list(read_corpus('a.csv'))
    train1 = train_corpus + train
    model = gensim.models.doc2vec.Doc2Vec(size=50, min_count=2, iter=55)
    model.build_vocab(train1)
    model.train(train1, total_examples=model.corpus_count, epochs=model.iter) 

    
    #fac_file =''
    X = []
    Y = []
    with open(op_file, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='|', quotechar='|')
        for row in spamreader:
	    print row
	    text = str(row[0])
	    sentence = gensim.utils.simple_preprocess(text)
	    a = model.infer_vector(sentence)
	    X.append(a)
	    Y.append(1)
    with open('a.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='|', quotechar='|')
        for row in spamreader:
            print row
     	    if(len(row)==0):
		continue
            text = str(row[0])
            sentence = gensim.utils.simple_preprocess(text)
            a = model.infer_vector(sentence)
            X.append(a)
            Y.append(0)
    x = np.array(X)
    y = np.array(Y)
    clf = GaussianNB()
    clf.fit(x,y)
    print "trained"
