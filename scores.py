import numpy as np
from tabulate import tabulate

from scipy.stats import ttest_rel
from scenarios import SCENARIOS, METRICS

def get_scores_table(path):
    scores = np.load(path)
    
    for s,scenario in enumerate(SCENARIOS):

        print(scenario)
        table = tabulate(scores[s],
                        tablefmt="grid",
                        headers=[i + 1 for i in range(len(scores[s][0]))],
                        showindex=[m for m in METRICS],
        )

        print(table)

        # table = tabulate(np.std(scores, axis=-1), 
        #                 tablefmt="grid", 
        #                 headers=["Accuracy", "BAC"],
        #                 showindex=["TRCR", "TRCP", "TSCP", "TSCS"]
        # )

        # print(table)


    # for each dataset
    # for i in range(scores.shape[0]):
    #     stat_mat = np.zeros((scores.shape[1], scores.shape[1]))
    #     for j in range(scores.shape[1]):
    #         for k in range(scores.shape[1]):
    #             t, p = ttest_rel(scores[i, j, :], scores[i, k, :])
    #             stat_mat[j, k] = p < 0.05
        
    #     table = tabulate(stat_mat)
    #     print(table)
