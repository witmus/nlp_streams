from LogisticRegressionWrapper import LogisticRegressionWrapper
from experiment import experiment

from TfidfTransformer import TfidfTransformer
from StreamClassifier import StreamClassifier

def logistic_regression_experiment():
    chunks = [(250,200),(500, 100), (750, 66), (1000, 50)]
    # chunks = [(10,10)]

    clfs = [
        StreamClassifier(TfidfTransformer(), LogisticRegressionWrapper(), is_transformer_refit=False, classifier_type='single'),
        StreamClassifier(TfidfTransformer(), LogisticRegressionWrapper(), is_transformer_refit=False, classifier_type='partial'),
        StreamClassifier(TfidfTransformer(), LogisticRegressionWrapper(), is_transformer_refit=True, classifier_type='refit'),
        StreamClassifier(TfidfTransformer(), LogisticRegressionWrapper(), is_transformer_refit=True, classifier_type='partial')
    ]

    experiment_name = "LogisticRegression"

    for c in chunks:
        experiment(c[0], c[1], clfs, experiment_name)
