import signal_processor as sp
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    y = sp.read_data(sp.SIGNAL_PATH)

    # trim data to size
    size = 1000000
    y_trimmed = y[:size]
    spikes = sp.find_spikes(y_trimmed)
    # todo: find mean distances

    x = np.arange(1, len(y_trimmed) + 1)
    plt.plot(x, y_trimmed)
    plt.xlabel("Próbka")
    plt.ylabel("Amplituda")
    plt.title(f'Sygnał: {size} próbek')
    plt.show()
