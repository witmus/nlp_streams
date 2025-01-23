from sklearn.linear_model import SGDClassifier
from experiment import experiment
from scores import get_scores_table

from TfidfTransformer import TfidfTransformer
from StreamClassifier import StreamClassifier

def sgdsvm_l2_experiment():
    chunks = [(250,200),(500, 100), (750, 66), (1000, 50), (1500,33), (2000, 25)]
    # chunks = [(10,10),(20,10)]

    experiment_name = "SGDSVM_L2"

    for c in chunks:#     
        clfs = [
            StreamClassifier(TfidfTransformer(), SGDClassifier(penalty='l2'), is_transformer_refit=False, classifier_type='single'),
            StreamClassifier(TfidfTransformer(), SGDClassifier(penalty='l2'), is_transformer_refit=False, classifier_type='partial'),
            StreamClassifier(TfidfTransformer(), SGDClassifier(penalty='l2'), is_transformer_refit=True, classifier_type='refit'),
            StreamClassifier(TfidfTransformer(), SGDClassifier(penalty='l2'), is_transformer_refit=True, classifier_type='partial')
        ]
        experiment(c[0], c[1], clfs, experiment_name)
