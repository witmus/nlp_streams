from sklearn.linear_model import SGDClassifier
from strlearn.streams import CSVParser
from strlearn.metrics import balanced_accuracy_score, specificity, recall
from strlearn.evaluators import TestThenTrain
from sklearn.metrics import accuracy_score
import numpy as np

def experiment(chunk_size, n_chunks, clfs, results_filename):
    stream = CSVParser('dataset_converted.csv', chunk_size=chunk_size, n_chunks=n_chunks)
    metrics = [accuracy_score, balanced_accuracy_score, specificity, recall]

    evaluator = TestThenTrain(metrics, verbose=True)
    evaluator.process(stream, clfs)

    scores = np.zeros(shape=(len(clfs), len(metrics), n_chunks - 1))
    for c, clf in enumerate(clfs):
        for m, metric in enumerate(metrics):
            for i in range(n_chunks - 1):
                scores[c,m,i] = evaluator.scores[c, i, m]
    
    np.save(f"scores/{results_filename}_scores_{chunk_size}_{n_chunks}", scores)
    