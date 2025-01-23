from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA

from TfidfTransformer import TfidfTransformer
from preprocessing import normalize_ndarray
import numpy as np

class StreamClassifier(BaseEstimator):
    def __init__(self, vectorizer: TfidfTransformer, classifier: BaseEstimator, is_transformer_refit : bool = False, classifier_type : str = 'partial'):
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.is_transformer_refit = is_transformer_refit
        self.classifier_type = classifier_type
        self.is_fit = False
        self.pca = PCA()
    
    def __str__(self):
        return "StreamClassifier"

    def fit(self, X, y):
        corpus = normalize_ndarray(X)
        vectors = np.array(self.vectorizer.fit_transform(corpus))
        X_train = self.pca.fit_transform(vectors)
        self.classifier.fit(X_train, y)
        self.is_fit = True
        return self

    def partial_fit(self, X, y, classes):
        if self.is_fit:
            if self.is_transformer_refit:
                corpus = normalize_ndarray(X)
                vectors = np.array(self.vectorizer.fit_transform(corpus))
                X_train = self.pca.fit_transform(vectors)
                if self.classifier_type == 'refit':
                    print('refitting clf')
                    self.classifier.fit(X_train, y)
                elif self.classifier_type == 'partial':
                    print('partial fitting clf on refitted transformer')
                    self.classifier.partial_fit(X_train, y, classes)
            else:
                if self.classifier_type == 'partial':
                    print('partial fitting clf on persistent transformer')
                    corpus = normalize_ndarray(X)
                    vectors = np.array([self.vectorizer.transform(c) for c in corpus])
                    X_train = self.pca.transform(vectors)
                    self.classifier.partial_fit(X_train, y, classes)
        else:
            return self.fit(X, y)
        return self

    def predict(self, X):
        corpus = normalize_ndarray(X)
        vectors = np.array([self.vectorizer.transform(c) for c in corpus])
        X_pred = self.pca.transform(vectors)
        return self.classifier.predict(X_pred)
