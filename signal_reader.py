from matplotlib import pyplot as plt
from scipy.stats import norm
import numpy as np

SIGNAL_PATH = 'data/signal_50MHz.bin'
SPIKE_THRESHOLD = 0.01
SPIKE_RISE = 50
SPIKE_FALL = 200

def read_data(path):
    with open(path, 'rb') as f:
        y = np.fromfile(f, dtype=np.float32)
    return y

def remove_spikes(input_array):
    spike_indices = np.where(input_array > SPIKE_THRESHOLD)[0]
    indices_to_remove = []
    for spike_index in spike_indices:
        indices_to_remove.extend(range(max(0, spike_index - SPIKE_RISE), min(len(input_array), spike_index + SPIKE_FALL)))
    indices_to_remove = list(set(indices_to_remove))
    cleaned_array = np.delete(input_array, indices_to_remove)
    return cleaned_array

def detect_spikes(input_array):
    spike_indices = np.where(input_array > SPIKE_THRESHOLD)[0]
    spikes = []
    distances = []

    start = 0
    end = 0
    for i, idx in enumerate(spike_indices):
        if i > 0 and idx - spike_indices[i-1] > 10:
            spikes.append(spike_indices[i-1])

    for i, idx in enumerate(spikes):
        if i > 0:
            distances.append(idx - spikes[i-1])

    return spikes, distances


if __name__ == "__main__":

    y = read_data(SIGNAL_PATH)

    # all data
    x = np.arange(1, len(y) + 1)
    plt.plot(x, y)
    plt.xlabel("Próbka")
    plt.ylabel("Amplituda")
    plt.title(f'Sygnał: {len(y)} próbek')
    plt.show()

    # trim data to size
    size = 1000000
    y_trimmed = y[:size]
    x = np.arange(1, len(y_trimmed) + 1)
    plt.plot(x, y_trimmed)
    plt.xlabel("Próbka")
    plt.ylabel("Amplituda")
    plt.title(f'Sygnał: {size} próbek')
    plt.show()

    # remove spikes
    y_no_spikes = remove_spikes(y_trimmed)
    x = np.arange(1, len(y_no_spikes) + 1)
    plt.plot(x, y_no_spikes)
    plt.xlabel("Próbka")
    plt.ylabel("Amplituda")
    plt.title(f'Sygnał: usunięte impulsy')
    plt.show()

    # histogram
    plt.hist(y_no_spikes, bins=50, density=True, alpha=0.6)
    mu, std = norm.fit(y_no_spikes)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    plt.xlabel("Amplituda")
    plt.ylabel("Ilość próbek")
    plt.title(f'Histogram amplitud dla {size} próbek')
    plt.show()

    # statistics
    print(f'Noise mean: {mu}')
    print(f'Noise standard deviation: {std}')

    # cumulative distribution function
    sorted_array = np.sort(y_no_spikes)
    cdf = np.arange(1, len(sorted_array) + 1) / len(sorted_array)
    plt.plot(sorted_array, cdf, label='CDF')
    plt.title('Dystrybuanta szumu')
    plt.xlabel('Wartość szumu')
    plt.ylabel('Prawdopodobieństwo')
    plt.grid(True)
    plt.show()

    # boxplot
    plt.boxplot(y_no_spikes)
    plt.grid(True)
    plt.show()

    # single spike
    y_spike = y[67650:67900]
    x_spike = np.arange(1, len(y_spike) + 1)
    plt.plot(x_spike, y_spike)
    plt.show()

    # marked spike threshold
    x = np.arange(1, len(y_trimmed) + 1)
    plt.plot(x, y_trimmed)
    plt.axhline(y = SPIKE_THRESHOLD, color = 'r', linestyle = '-')
    plt.xlabel("Próbka")
    plt.ylabel("Amplituda")
    plt.title(f'Linia odcięcia impulsów')
    plt.show()

    # detect spikes
    spikes, distances = detect_spikes(y)
    print(f'Total spikes detected: {len(spikes)}')
    print(f'Mean distance between spikes: {np.mean(distances)}')
    print(f'Min distance between spikes: {np.min(distances)}')
    print(f'Max distance between spikes: {np.max(distances)}')
