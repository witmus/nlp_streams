from random import random
import matplotlib.pyplot as plt
import numpy as np
from scenarios import SCENARIOS, METRICS

def plot_by_scenarios(classifier: str, chunk_size: int, n_chunks: int):
    scores = np.load(f'scores/{classifier}_scores_{chunk_size}_{n_chunks}.npy')
    fig,axs = plt.subplots(2,2, figsize=(16,9), layout='constrained')
    plt.title(f'{classifier}\nchunk_size={chunk_size} n_chunks={n_chunks}')
    for ax, (m,metric) in zip(axs.flat, enumerate(METRICS)):
        ax.set_ylim((0,1))
        ax.set_xlabel('Chunk')
        ax.set_ylabel(metric)
        for s,scenario in enumerate(SCENARIOS):
            ax.plot(scores[s,m,:], label=scenario)
            ax.legend()

    plt.savefig(f'plots_methods/{classifier}/{classifier}_{chunk_size}_{n_chunks}.png')
    plt.close()
