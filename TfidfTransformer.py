import math
from typing import List
import pandas as pd
from sklearn.feature_extraction.text import _VectorizerMixin
from sklearn.base import BaseEstimator

class TfidfTransformer(BaseEstimator, _VectorizerMixin):
    def __init__(self):
        self.ndocs : int = 0
        self.df : dict[str,int] = dict()
        self.idf : dict[str,float] = dict()

    def fit(self, X: List[List[str]], y = None):
        self.ndocs = len(X)

        for d in X:
            words = list(dict.fromkeys(d))
            for w in words:
                if w in self.df:
                    self.df[w] += 1
                else:
                    self.df.update({w:1})
        
        N = len(X)
        for w,df in self.df.items():
            idf = math.log10(N / float(df + 1))
            self.idf.update({w:idf})
        
        return self
    
    def get_idf(self):
        return self.idf
    
    def transform(self, document: List[str]) -> List[float]:
        documentLength = len(document)
        tfidfs = dict.fromkeys(self.idf.keys(), 0.0)
        wordsOccurences = dict.fromkeys(document, 0)
        for w in document:
            wordsOccurences[w] += 1

        for w,o in wordsOccurences.items():
            if w in self.idf:
                idf = self.idf[w]
                tf = o / float(documentLength)
                tfidfs[w] = tf * idf
        
        return list(tfidfs.values())
    
    def fit_transform(self, X: List[List[str]], y = None) -> List[List[float]]:
        self.fit(X,y)
        result = []
        for x in X:
            result.append(self.transform(x))
        return result

