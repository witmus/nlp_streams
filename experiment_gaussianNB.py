from sklearn.naive_bayes import GaussianNB
from experiment import experiment

from TfidfTransformer import TfidfTransformer
from StreamClassifier import StreamClassifier

def gaussianNB_experiment():
    chunks = [(250,200),(500, 100), (750, 66), (1000, 50), (1500,33), (2000, 25)]
    # chunks = [(10,10),(20,10)]

    experiment_name = "GaussianNB"

    for c in chunks:
        clfs = [
            StreamClassifier(TfidfTransformer(), GaussianNB(), is_transformer_refit=False, classifier_type='single'),
            StreamClassifier(TfidfTransformer(), GaussianNB(), is_transformer_refit=False, classifier_type='partial'),
            StreamClassifier(TfidfTransformer(), GaussianNB(), is_transformer_refit=True, classifier_type='refit'),
            StreamClassifier(TfidfTransformer(), GaussianNB(), is_transformer_refit=True, classifier_type='partial')
        ]
        experiment(c[0], c[1], clfs, experiment_name)
