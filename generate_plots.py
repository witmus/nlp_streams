from plot_scores import plot_scores

clfs = [
    'GaussianNB',
    'SGDSVM_L1',
    'SGDSVM_L2',
    'LogisticRegression'
]

chunks = [
    (250,200),
    (500, 100),
    (750, 66),
    (1000, 50),
    (1500,33),
    (2000, 25)
]

for clf in clfs:
    for c in chunks:
        plot_scores(clf,c[0],c[1])
