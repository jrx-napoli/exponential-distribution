import numpy as np
from matplotlib import pyplot as plt

import signal_reader as sr

if __name__ == "__main__":
    y = sr.read_data(sr.SIGNAL_PATH)

    # all data
    # x = np.arange(1, len(y) + 1)
    # plt.plot(x, y)
    # plt.xlabel("Próbka")
    # plt.ylabel("Amplituda")
    # plt.title(f'Sygnał: {len(y)} próbek')
    # plt.show()

    # trim data to size
    size = 100000
    y_trimmed = y[:size]
    peaks = sr.find_spikes(y_trimmed)

    x = np.arange(1, len(y_trimmed) + 1)
    plt.plot(x, y_trimmed)
    plt.xlabel("Próbka")
    plt.ylabel("Amplituda")
    plt.title(f'Sygnał: {size} próbek')
    plt.show()
