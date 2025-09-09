# models.py

from sentiment_data import *
from utils import *

from collections import Counter

import numpy as np
import random
from typing import List, Tuple

import nltk
from nltk.corpus import stopwords

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    Converts words from a sentence into numbers that a machine learning model can understand.
    Just defines the interface, the real work is done in subclasses.

    Attributes:
        get_indexer: returns the Indexer used by this feature extractor
        extract_features: extracts features from a sentence
    """
    def get_indexer(self):
        """
        :return: the Indexer used by this feature extractor
        """
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.

    Uses single words as features.
    Don't worry about bias.
    """
    def __init__(self, indexer: Indexer, lowercase: bool=True, binary: bool=True):
        """
        :param indexer: Indexer object to use for this featurizer
        :param lowercase: if True, lowercase all words before adding to indexer
        :param binary: if True, use binary features (0/1) instead of counts
        """
        self.indexer = indexer
        self.lowercase = lowercase
        self.binary = binary
    
    def get_indexer(self):
        return self.indexer
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        features = Counter()
        for word in sentence:
            if self.lowercase:
                word = word.lower()
            if add_to_indexer:
                index = self.indexer.add_and_get_index(word)
            else:
                index = self.indexer.index_of(word)   # FIXED: use index_of instead of get_index
            if index != -1:
                if self.binary:
                    features[index] = 1
                    # print(f"[UNIGRAM] feature='{word}' idx={index} val={features[index]}")
                else:
                    features[index] += 1
                    # print(f"[UNIGRAM] feature='{word}' idx={index} val={features[index]}")
        # print(f"[UNIGRAM] total_active_featuress={len(features)} | vocab_size={len(self.indexer)}")
        return features




class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    Don't worry about bias.
    """
    def __init__(self, indexer: Indexer, lowercase: bool=True, binary: bool=True):
        """
        :param indexer: Indexer object to use for this featurizer
        :param lowercase: if True, lowercase all words before adding to indexer
        :param binary: if True, use binary features (0/1) instead of counts
        """
        self.indexer = indexer
        self.lowercase = lowercase
        self.binary = binary
    
    def get_indexer(self):
        return self.indexer
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        features = Counter()
        for i in range(len(sentence) - 1):
            word1 = sentence[i]
            word2 = sentence[i + 1]
            if self.lowercase:
                word1 = word1.lower()
                word2 = word2.lower()
            bigram = f"BIGRAM_{word1}_{word2}"
            if add_to_indexer:
                index = self.indexer.add_and_get_index(bigram)
            else:
                index = self.indexer.index_of(bigram)
            if index != -1:
                if self.binary:
                    features[index] = 1
                    # print(f"[BIGRAM] feature='{bigram}' idx={index} val={features[index]}")
                else:
                    features[index] += 1
                    # print(f"[BIGRAM] feature='{bigram}' idx={index} val={features[index]}")
        # print(f"[BIGRAM] total_active_features={len(features)} | vocab_size={len(self.indexer)}")
        return features


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!

    Don't worry about bias.
    Uses unigrams (except stopwords) and bigrams (also skipping stopwords) as features.
    """
    def __init__(self, indexer: Indexer, lowercase: bool=True, binary: bool=True):
        self.indexer = indexer
        self.lowercase = lowercase
        self.binary = binary
        self.stop_words = set(stopwords.words('english'))
    
    def get_indexer(self):
        return self.indexer
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        features = Counter()

        if len(sentence) == 0:
            return features
        
        # Unigram features (excluding stopwords)
        for word in sentence:
            if self.lowercase:
                word = word.lower()
            if word in self.stop_words:
                continue
            if add_to_indexer:
                index = self.indexer.add_and_get_index(word)
            else:
                index = self.indexer.index_of(word)
            if index != -1:
                if self.binary:
                    features[index] = 1
                    # print(f"[BETTER-UNIGRAM] feature='{word}' idx={index} val={features[index]}")
                else:
                    features[index] += 1
                    # print(f"[BETTER-UNIGRAM] feature='{word}' idx={index} val={features[index]}")
        
        # Bigram features (excluding stopwords)
        for i in range(len(sentence) - 1):
            word1 = sentence[i]
            word2 = sentence[i + 1]
            if self.lowercase:
                word1 = word1.lower()
                word2 = word2.lower()
            if word1 in self.stop_words or word2 in self.stop_words:
                continue
            bigram = f"{word1}_{word2}"
            if add_to_indexer:
                index = self.indexer.add_and_get_index(bigram)
            else:
                index = self.indexer.index_of(bigram)
            if index != -1:
                if self.binary:
                    features[index] = 1
                    # print(f"[BETTER-BIGRAM] feature='{bigram}' idx={index} val={features[index]}")
                else:
                    features[index] += 1
                    # print(f"[BETTER-BIGRAM] feature='{bigram}' idx={index} val={features[index]}")
        # print(f"[BETTER] total_active_features={len(features)} | vocab_size={len(self.indexer)}")
        return features



class SentimentClassifier(object):
    """
    Sentiment classifier base type
    Every classifier must implement the predict method, which takes a list of words and returns either 0 or 1 to label the sentence as negative or positive.
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    Exists as an example, provides baseline
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.

    Stores a weight vector (learned parameters) and a feature extractor (to convert sentences into feature vectors).
    Uses the feature extractor to convert a sentence into indices.
    Computes the dot product of the weight vector with the feature vector.

    Attributes:
        weights: numpy array of weights for the perceptron
        feat_extractor: feature extractor to use
    """
    def __init__(self, weights: np.ndarray, feat_extractor: FeatureExtractor):
        self.weights = weights
        self.feat_extractor = feat_extractor
    
    def predict(self, sentence: List[str]) -> int:
        features = self.feat_extractor.extract_features(sentence, add_to_indexer=False)
        score = sum(self.weights[idx] * value for idx, value in features.items())
        # print(f"[PRED:PERCEPTRON] score={score:.4f} active_feats={len(feats)}")
        return 1 if score >= 0 else 0


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.

    Predicts with the sigmoid function applied to the dot product of the weight vector and feature vector.
    """
    def __init__(self, weights: np.ndarray, feat_extractor: FeatureExtractor):
        """
        :param weights: numpy array of weights for the logistic regression model
        :param feat_extractor: feature extractor to use
        """
        self.weights = weights
        self.feat_extractor = feat_extractor
    
    def predict(self, sentence: List[str]) -> int:
        """
        Predicts the label (0 or 1) for a given sentence using logistic regression (sigmoid of dot product).
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        features = self.feat_extractor.extract_features(sentence, add_to_indexer=False)
        score = sum(self.weights[idx] * value for idx, value in features.items())
        prob = 1 / (1 + np.exp(-score))
        # print(f"[PRED:LR] score={score:.4f} prob={prob:.4f} active_feats={len(feats)}")
        return 1 if prob >= 0.5 else 0


def train_perceptron(
    train_exs: List[SentimentExample],
    feat_extractor: FeatureExtractor,
    *,
    num_epochs: int = 5,
    lr: float = 1.0
) -> PerceptronClassifier:
    """
    Train a perceptron model.
    Loops over the training examples for a set number of times (epochs).
    For each example, it extracts features, computes the prediction, and updates the weights if the prediction is incorrect.

    y = label
    p = predicted label
    For each feature i:
        weights[i] += learning_rate * (y - p) * x[i]
    
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :param num_epochs: number of epochs to train for
    :param lr: learning rate for weight updates
    :return: trained PerceptronClassifier model
    """
    # build the indexer with all of the features
    for ex in train_exs:
        feat_extractor.extract_features(ex.words, add_to_indexer=True)
    
    weights = np.zeros(len(feat_extractor.get_indexer()))
    # print(f"[TRAIN:PERCEPTRON] vocab_size={len(feat_extractor.get_indexer())} epochs={num_epochs} lr={lr}")
    for epoch in range(num_epochs):
        random.shuffle(train_exs)
        for ex in train_exs:
            features = feat_extractor.extract_features(ex.words, add_to_indexer=False)
            score = sum(weights[idx] * value for idx, value in features.items())
            prediction = 1 if score >= 0 else 0
            if prediction != ex.label:
                for idx, value in features.items():
                    weights[idx] += lr * (ex.label - prediction) * value
    
    # print(f"[PERCEPTRON][ep={ep}] mistakes={mistakes} total={len(train_exs)} err_rate={mistakes/len(train_exs):.3f}")
    return PerceptronClassifier(weights, feat_extractor)


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor,
                              num_epochs: int = 10, learning_rate: float = 0.1) -> LogisticRegressionClassifier:
    """
    Using Stochastic Gradient Descent
    Loops over the training examples for a set number of times (epochs).
    For each example, it extracts features, computes the prediction with sigmoid, and updates the weights if the prediction is incorrect.

    y = label
    p = predicted probability
    For each feature i:
        weights[i] += learning_rate * (y - p) * x[i]
    


    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :param num_epochs: number of epochs to train for
    :param learning_rate: learning rate for weight updates
    :return: trained LogisticRegressionClassifier model
    """
    # build the indexer with all of the features
    for ex in train_exs:
        feat_extractor.extract_features(ex.words, add_to_indexer=True)
    
    weights = np.zeros(len(feat_extractor.get_indexer()))
    # print(f"[TRAIN:LR] vocab_size={len(feat_extractor.get_indexer())} epochs={num_epochs} lr={learning_rate}")
    for epoch in range(num_epochs):
        random.shuffle(train_exs)
        for ex in train_exs:
            features = feat_extractor.extract_features(ex.words, add_to_indexer=False)
            score = sum(weights[idx] * value for idx, value in features.items())
            prob = 1 / (1 + np.exp(-score))
            for idx, value in features.items():
                weights[idx] += learning_rate * (ex.label - prob) * value
    
    # avg_ll = running_ll / len(train_exs)
    # print(f"[LR][ep={ep}] avg_log_likelihood={avg_ll:.4f}")
    
    return LogisticRegressionClassifier(weights, feat_extractor)


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model