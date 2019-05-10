from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import *
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from analysis_and_split import get_items_in_directory
from datetime import datetime
from nltk.util import ngrams
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.metrics import BigramAssocMeasures
from operator import itemgetter
import re

def removeURL(comment):
    """Removes URLs from comments"""
    return re.sub(r'http\S+', '', comment)


def get_features(type):
    output_labels = []
    output_documents = []
    words = defaultdict(list)
    labels = get_items_in_directory('directory',type)
    for team in labels:
        files = get_items_in_directory('file',type + '/' + team)
        count = 0
        for file in files:
            data = open(type + '/' + team + '/' + file).read()
            comments = data.split("##########")
            usable_comments = []
            for comment in comments:
                comment = removeURL(comment)
                tok_comment = RegexpTokenizer(r'\w+').tokenize(comment.strip().lower())
                if len(tok_comment) > 0:
                    for word in tok_comment:
                        words[team].append(word)
                    usable_comments.append(tok_comment)
            output_documents.append([inner for outer in usable_comments for inner in outer])
            output_labels.append(team)
            count += 1
            #if count == 5:
                #break
    return output_documents,output_labels, words


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
        high_info_words |= set([word for word, score in bestwords])

    return high_info_words


def high_information(comments,labels,words,n):
    labels_set = set(labels)
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
    Xtrain, Ytrain, words = get_features('train')
    Xtest, Ytest, words_not_used = get_features('test')
    n = 100
    high_info_words = high_information(Xtrain,Ytrain,words,n)
    HIXtrain, HIYtrain = filter_high_info_and_stop_words(Xtrain,Ytrain,high_info_words)
    #HIXtest, HIYtest = filter_high_info_and_stop_words(Xtest,Ytest,high_info_words)
    vec = CountVectorizer(preprocessor=identity,tokenizer=identity)
    clf = MultinomialNB()
    #clf = LinearSVC()
    #clf = LogisticRegression(multi_class='ovr')
    classifier = Pipeline([('vec',vec),('clf',clf)])
    classifier.fit(HIXtrain,HIYtrain)
    Yguess = classifier.predict(Xtest)
    for item in list(set(HIYtrain)):
        show_most_informative_features(vec,clf,item)
    print("Accuracy:{}".format(accuracy_score(Ytest,Yguess)))
    print(classification_report(Ytest,Yguess))
    print("Runtime: {} seconds.".format(datetime.now()-startTime))
if __name__ == "__main__":
    main()
