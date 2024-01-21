import signal_reader as sr
import numpy as np
from matplotlib import pyplot as plt

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
    size = 1000000
    y_trimmed = y[:size]
    x = np.arange(1, len(y_trimmed) + 1)

    plt.plot(x, y_trimmed)
    plt.xlabel("Próbka")
    plt.ylabel("Amplituda")
    plt.title(f'Sygnał: {size} próbek')
    plt.show()
