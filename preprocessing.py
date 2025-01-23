import re
import numpy as np
from typing import List
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords

# tokenizes and stems input dataframe, removes HTML tags and stopwords
# returns corpus as 2D array 
def normalize_dataframe(df: pd.DataFrame) -> List[List[str]]:
    result = []
    sw = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    for _, row in df.iterrows():
        review = re.sub('[^a-zA-Z]', ' ', row['review'])
        review = review.lower()
        review = review.split()
        review = [stemmer.stem(w) for w in review if w not in sw]
        result.append(review)
    
    return result

def normalize_ndarray(arr: np.ndarray) -> List[List[str]]:
    result = []
    sw = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    for a in arr:
        review = a[0]
        review = re.sub('<[^>]*>', ' ', review)
        review = re.sub('[^a-zA-Z]', ' ', review)
        review = review.lower()
        review = review.split()
        review = [stemmer.stem(w) for w in review if w not in sw]
        result.append(review)
    
    return result
