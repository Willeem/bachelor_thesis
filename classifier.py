from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import *
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from os import listdir, makedirs, getcwd
from os.path import isdir, isfile, join
from datetime import datetime
from nltk.util import ngrams

from nltk.corpus import stopwords
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.metrics import BigramAssocMeasures
from operator import itemgetter
import numpy as np

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
                with open(type + '/' + team + '/' + file) as f:
                    data = f.read()
                    feats = data.split()
                for word in feats:
                    words[team].append(word)
                features.append(feats)
                output_labels.append(team)
                count += 1
                #if count == 5:
                    #break
    return features,output_labels, words


def filter_high_info_and_stop_words(comments,labels,high_info_words):
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

def high_information_words(labelled_words, score_fn=BigramAssocMeasures.chi_sq,n=5):
    word_fd = FreqDist()
    label_word_fd = ConditionalFreqDist()

    for label, words in labelled_words:
        for word in words:
            word_fd[word] += 1
            label_word_fd[label][word] += 1

    n_xx = label_word_fd.N()
    high_info_words = set()

    for label in label_word_fd.conditions():
        n_xi = label_word_fd[label].N()
        word_scores = defaultdict(int)

        for word, n_ii in label_word_fd[label].items():
            n_ix = word_fd[word]
            score = score_fn(n_ii, (n_ix, n_xi), n_xx)
            word_scores[word] = score
        bestwords = sorted(word_scores.items(), key=itemgetter(1),reverse=True)[:n]
        print(label,"& \\textit{"," ".join([word for word, score in bestwords[:10]]),"} \\\\")
        high_info_words |= set([word for word, score in bestwords])

    return high_info_words


def high_information(comments,labels,words,n):
    labels_set = set(labels)
    labels_set = sorted(labels_set)
    all_words = [word for comment in comments for word in comment]
    labelled_words = [(label,words[label]) for label in labels_set]
    high_info_words = set(high_information_words(labelled_words,n=n))
    print("Number of words in the data: %i" % len(all_words))
    print("Number of distinct words in the data: %i" % len(set(all_words)))
    print("Number of distinct high info words in the data: %i" % len(high_info_words))
    return high_info_words

def show_most_informative_features(vectorizer, classifier, class_label, n=20):
    labelid = list(classifier.classes_).index(class_label)
    feature_names = vectorizer.get_feature_names()
    topn = sorted(zip(classifier.coef_[labelid],feature_names))[-n:]

    for coef, feat in topn:
        print(class_label,feat,coef)


def main():
    startTime = datetime.now()
    Xtrain, Ytrain, words = get_features(['train','dev'])
    Xtest, Ytest, words_not_used = get_features(['test'])
    n = 100
    #for n in [10,50,100,500,1000,5000,10000]:
    high_info_words = high_information(Xtrain,Ytrain,words,n)
    HIXtrain, HIYtrain = filter_high_info_and_stop_words(Xtrain,Ytrain,high_info_words)
    vec = CountVectorizer(preprocessor=identity,tokenizer=identity)
    clf = MultinomialNB()
    #clf = LinearSVC()
    #clf = LogisticRegression(multi_class='ovr')
    classifier = Pipeline([('vec',vec),('clf',clf)])
    classifier.fit(HIXtrain,HIYtrain)
    Yguess = classifier.predict(Xtest)
        #for item in list(set(HIYtrain)):
    #    show_most_informative_features(vec,clf,item)
    print("Accuracy:{}".format(accuracy_score(Ytest,Yguess)))
    print(classification_report(Ytest,Yguess))
    misclassified = np.where(Ytest != Yguess)
    for item in np.nditer(misclassified):
        print(Ytest[item],Yguess[item])
    print("Runtime: {} seconds.".format(datetime.now()-startTime))
if __name__ == "__main__":
    main()
