import matplotlib.pyplot as plt
import numpy as np
from scenarios import SCENARIOS, METRICS

def plot_scores(classifier: str, chunk_size: int, n_chunks: int):
    scores = np.load(f'scores/{classifier}_scores_{chunk_size}_{n_chunks}.npy')
    for s,scenario in enumerate(SCENARIOS):
        plt.figure(figsize=(12,8))
        
        plt.subplot(211)
        plt.title(f'{classifier}\n{scenario}\nchunk_size={chunk_size} n_chunks={n_chunks}')
        plt.xlabel('Chunk')
        plt.ylabel('Metric')
        plt.ylim(0,1)
        plt.plot(scores[s,0,:], label=METRICS[0])
        plt.plot(scores[s,1,:], label=METRICS[1])
        plt.legend()

        plt.subplot(212)
        plt.xlabel('Chunk')
        plt.ylabel('Metric')
        plt.ylim(0,1)
        plt.plot(scores[s,2,:], label=METRICS[2])
        plt.plot(scores[s,3,:], label=METRICS[3])
        plt.legend()
        plt.savefig(f'plots/{classifier}/{classifier}_{scenario}_{chunk_size}_{n_chunks}.png')
        plt.close()
