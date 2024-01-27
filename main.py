import signal_processor as sp
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import expon

MAX_DATA_SIZE = 50_000_000

if __name__ == "__main__":
    y = sp.read_data(sp.SIGNAL_PATH)

    # trim data to size
    size = MAX_DATA_SIZE
    y_trimmed = y[:size]

    # get the positions of spikes
    spikes = sp.find_spikes(y_trimmed)

    # calculate distances
    distances = sp.calculate_distances(spikes)

    # estimate lambda
    lambda_est = 1 / np.mean(distances)

    # Narysuj histogram
    plt.hist(distances, bins=30, density=True, alpha=0.7, label='Histogram')

    # Narysuj funkcję gęstości prawdopodobieństwa rozkładu wykładniczego
    x = np.linspace(0, np.max(distances), 100)
    y = expon.pdf(x, scale=1 / lambda_est)
    plt.plot(x, y, 'r-', label='Exp($\lambda$)')

    # Dodaj legendę, etykiety itp.
    plt.legend()
    plt.xlabel('Odstęp czasowy')
    plt.ylabel('Prawdopodobieństwo')
    plt.show()

    # x = np.arange(1, len(y_trimmed) + 1)
    # plt.plot(x, y_trimmed)
    # plt.xlabel("Próbka")
    # plt.ylabel("Amplituda")
    # plt.title(f'Sygnał: {size} próbek')
    # plt.show()
