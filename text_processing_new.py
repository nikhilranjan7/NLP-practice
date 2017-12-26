import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import word_tokenize
import pickle

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier

from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

short_pos = open("data/pos.txt", "r").read()
short_neg = open("data/neg.txt", "r").read()

document = []

for r in short_pos.split('\n'):
    document.append((r,"pos"))

for r in short_neg.split('\n'):
    document.append((r,"neg"))

random.shuffle(document)
all_words = []

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
print(all_words.most_common(15))
print(all_words["blast"])

#word_features = list(all_words.keys())[:3000]
word_features = [a[0] for a in all_words.most_common(5000)]

def find_features(document):
    words = set(word_tokenize(document))
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

#print(find_features(movie_reviews.words('neg/cv000_29416.txt')))

featuresets = [(find_features(rev), category) for (rev, category) in document] # same as documents but instead of words list only will give a dictionary to represent whether the word is in top 1000 word or not
random.shuffle(featuresets)

training_set = featuresets[:10000]
testing_set = featuresets[10000:]

#classifier = nltk.NaiveBayesClassifier.train(training_set)

classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

print("Original Naive Bayes Algo accuracy:", nltk.classify.accuracy(classifier, testing_set))
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy:", nltk.classify.accuracy(MNB_classifier, testing_set))

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy:", nltk.classify.accuracy(BernoulliNB_classifier, testing_set))

MLPClassifier = SklearnClassifier(MLPClassifier())
MLPClassifier.train(training_set)
print("MLPClassifier accuracy:", nltk.classify.accuracy(MLPClassifier, testing_set))

voted_classifier = VoteClassifier(classifier, MNB_classifier, BernoulliNB_classifier)
print("voted_classifier accuracy: ", (nltk.classify.accuracy(voted_classifier, testing_set)))

print("Classification:", voted_classifier.classify(testing_set[0][0]), "\n confidence :", voted_classifier.confidence(testing_set[0][0]))

# Check the documentation for NLTK classify method
