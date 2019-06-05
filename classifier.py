from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import *
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from os import listdir, makedirs, getcwd
from os.path import isdir, isfile, join
from datetime import datetime
from nltk.util import ngrams

from nltk import ngrams, pos_tag
from nltk.corpus import stopwords
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.metrics import BigramAssocMeasures
from operator import itemgetter
import numpy as np
import pickle

def get_items_in_directory(filetype,directory):
    """Lists all items in a directory depending on filetype (directory or file)"""
    if filetype == 'directory':
        return [f for f in listdir(directory) if isdir(join(directory,f))]
    return [f for f in listdir(directory) if isfile(join(directory,f))]

def get_features(types):
    output_labels = []
    features = []
    words = defaultdict(list)
    for type in types:
        labels = get_items_in_directory('directory',type)
        for team in labels:
            files = get_items_in_directory('file',type + '/' + team)
            count = 0
            for file in files:
                feats = []
                with open(type + '/' + team + '/' + file) as f:
                    data = f.read()
                    for word in data.split():
                        feats.append(word)
                        words[team].append(word)
                    features.append(feats)
                    output_labels.append(team)
                count += 1
                if count == 5:
                    break
    return features,output_labels, words


def filter_high_info_and_stop_words(comments,labels,all_word_scores,n):
    high_info_words = set()
    for label in all_word_scores:
        for word_scores in all_word_scores[label]:
            bestwords = sorted(word_scores.items(), key=itemgetter(1),reverse=True)[:n]
            #print(label,"& \\textit{",[word for word, score in bestwords[:10]],"} \\\\")
            high_info_words |= set([word for word, score in bestwords])
    print("Number of distinct high info words in the data: %i" % len(high_info_words))

    output_labels = []
    output_documents = []
    stop_words = set(stopwords.words('english'))
    index = 0
    for comment in comments:
        filtered_comment = [word for word in comment if word not in stop_words and word in high_info_words]
        if len(filtered_comment) > 0:
            output_labels.append(labels[index])
            output_documents.append(filtered_comment)
        index += 1
    return output_documents, output_labels

def identity(x):
    return x

def high_information_words(labelled_words, score_fn=BigramAssocMeasures.chi_sq):
    word_fd = FreqDist()
    label_word_fd = ConditionalFreqDist()

    for label, words in labelled_words:
        for word in words:
            word_fd[word] += 1
            label_word_fd[label][word] += 1

    n_xx = label_word_fd.N()
    all_scores = defaultdict(list)
    for label in label_word_fd.conditions():
        n_xi = label_word_fd[label].N()
        word_scores = defaultdict(int)

        for word, n_ii in label_word_fd[label].items():
            n_ix = word_fd[word]
            score = score_fn(n_ii, (n_ix, n_xi), n_xx)
            word_scores[word] = score
        all_scores[label].append(word_scores)
    return all_scores


def high_information(comments,labels,words):
    labels_set = set(labels)
    sorted_labels = sorted(labels_set)
    #all_words = [word for comment in comments for word in comment]
    labelled_words = [(label,words[label]) for label in sorted_labels]
    word_scores = high_information_words(labelled_words)
    #print("Number of words in the data: %i" % len(all_words))
    #print("Number of distinct words in the data: %i" % len(set(all_words)))
    return word_scores

def ngramsplitter(x):
    return x.split()

def get_pos_tags(file):
    objs = []
    f = open(file,'rb')
    while 1:
        try:
            objs.append(pickle.load(f))
        except EOFError:
            break
    objs = [i for obj in objs for item in obj for i in item]
    return objs

def main():
    startTime = datetime.now()
    Xtrain, Ytrain, words = get_features(['train'])
    Xtest, Ytest, words_not_used = get_features(['dev'])

    Xtrain = get_pos_tags('pos_tags_train.pickle')
    Xtest = get_pos_tags('pos_tags_dev.pickle')


    print('read')
    # word_scores = high_information(Xtrain,Ytrain,words)
    #for n in [10,50,100,500,1000,5000,10000]:
    # n = 100
    # HIXtrain, HIYtrain = filter_high_info_and_stop_words(Xtrain,Ytrain,word_scores,n)
    # for item in HIXtrain:
    #     Xtrain.append(item)
    # for item in HIYtrain:
    #     Ytrain.append(item)
    #Xtrain = [str(x) for x in Xtrain]
    #Xtest = [str(x) for x in Xtest]
    tfidf = TfidfVectorizer(preprocessor=identity,tokenizer=identity)

    for i in [(2,2),(3,3),(4,4),(5,5)]:
        cvt = CountVectorizer(preprocessor=identity,tokenizer=identity,analyzer='char_wb',ngram_range=i)
        tfidf = TfidfVectorizer(preprocessor=identity,tokenizer=identity)#,analyzer='char_wb',ngram_range=i)
        tfidfngram = TfidfVectorizer(preprocessor=identity,tokenizer=identity,analyzer='char_wb',ngram_range=i)
        for combination in [(cvt,MultinomialNB()),(tfidfngram,MultinomialNB()),(cvt,LogisticRegression(multi_class='ovr')),(tfidfngram,LogisticRegression(multi_class='ovr'))]:
            vec = combination[0]
            clf = combination[1]
            #clf = LinearSVC()
            #vec = tfidf5grams
            #clf = LinearSVC()
            #clf = LogisticRegression(multi_class='ovr')
            #classifier = Pipeline([('vec',vec),('clf',clf)])
            #for i in [(2,2),(3,3),(4,4),(5,5)]:
            classifier = Pipeline([
            ('features', FeatureUnion([
                ('text', Pipeline([
                    ('vec',tfidf),
                ])),
                ('ngrams', Pipeline([
                    ('ngram',vec)
                ])),
                #('length', Pipeline([
                #    ('count',FunctionTransformer(get_word_length,validate=False))
                #])),
            ])),
            ('clf',clf)])
            classifier.fit(Xtrain,Ytrain)
            Yguess = classifier.predict(Xtest)
            print("i: {} Accuracy:{}".format(i,accuracy_score(Ytest,Yguess)))
            #print(classification_report(Ytest,Yguess))
    print("Runtime: {} seconds.".format(datetime.now()-startTime))
if __name__ == "__main__":
    main()
