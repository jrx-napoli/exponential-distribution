import signal_processor as sp
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import expon

SIGNAL_PATH = 'data/signal_50MHz.bin'

if __name__ == "__main__":
    y = sp.read_data(SIGNAL_PATH)

    # get total size of data
    total_size = len(y)

    # get the positions of spikes
    spikes = sp.find_spikes(y)

    # calculate distances
    distances = sp.calculate_distances(spikes)

    # convert distances to milliseconds
    for i in range(len(distances)):
        distances[i] = (distances[i] / total_size) * 1000

    # estimate lambda
    lambda_est = 1 / np.mean(distances)

    # draw histogram
    plt.hist(distances, bins=30, density=True, alpha=0.7, label='Histogram')

    # draw pdf
    x = np.linspace(0, np.max(distances), 100)
    y = expon.pdf(x, scale=1 / lambda_est)
    plt.plot(x, y, 'r-', label='Exp($\\lambda$)')

    # display on one figure
    plt.legend()
    plt.xlabel('Time difference [ms]')
    plt.ylabel('Probability')
    plt.show()
