from sklearn.linear_model import LogisticRegression

class LogisticRegressionWrapper(LogisticRegression):
    def __init__(self, warm_start=True):
        super().__init__(warm_start=warm_start)
    
    def fit(self, X, y):
        return super().fit(X,y)
    
    def partial_fit(self, X, y, classes):
        return super().fit(X,y)
    
    def predict(self, X):
        return super().predict(X)