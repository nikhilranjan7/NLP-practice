import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
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

document = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        document.append((list(movie_reviews.words(fileid)),category))   # will give something like [([words........], pos), ([words.....], neg)....]

random.shuffle(document)

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
print(all_words.most_common(15))
print(all_words["blast"])

#word_features = list(all_words.keys())[:3000]
word_features = [a[0] for a in all_words.most_common(3000)]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

#print(find_features(movie_reviews.words('neg/cv000_29416.txt')))

featuresets = [(find_features(rev), category) for (rev, category) in document] # same as documents but instead of words list only will give a dictionary to represent whether the word is in top 1000 word or not

training_set = featuresets[:1900]
testing_set = featuresets[1900:]

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
