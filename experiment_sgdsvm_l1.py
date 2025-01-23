from sklearn.linear_model import SGDClassifier
from experiment import experiment
from scores import get_scores_table

from TfidfTransformer import TfidfTransformer
from StreamClassifier import StreamClassifier

def sgdsvm_l1_experiment():
    chunks = [(250,200),(500, 100), (750, 66), (1000, 50)]
    # chunks = [(10,10)]

    clfs = [
        StreamClassifier(TfidfTransformer(), SGDClassifier(penalty='l1'), is_transformer_refit=False, classifier_type='single'),
        StreamClassifier(TfidfTransformer(), SGDClassifier(penalty='l1'), is_transformer_refit=False, classifier_type='partial'),
        StreamClassifier(TfidfTransformer(), SGDClassifier(penalty='l1'), is_transformer_refit=True, classifier_type='refit'),
        StreamClassifier(TfidfTransformer(), SGDClassifier(penalty='l1'), is_transformer_refit=True, classifier_type='partial')
    ]

    experiment_name = "SGDSVM_L1"

    for c in chunks:
        experiment(c[0], c[1], clfs, experiment_name)
