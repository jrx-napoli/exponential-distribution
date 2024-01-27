import numpy as np

SPIKE_THRESHOLD = 0.01
MAX_SPIKE_INCONSISTENCY = 200
SPIKE_RISE = 50
SPIKE_FALL = 200


def read_data(path):
    with open(path, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
    return data


def _find_peak_indexes(data, spikes_idx):
    """
    Finds the exact indexes of spikes' peaks

    :param data: (ndarray) Array of signal data
    :param spikes_idx: (list) List of lists of spikes indexes
    :return: (list) Indexes of local maximums for aLL spikes
    """
    peaks = []

    for spike_idx in spikes_idx:
        spike_max_idx = spike_idx[0]

        for idx in spike_idx:
            if data[idx] > data[spike_max_idx]:
                spike_max_idx = idx

        peaks.append(spike_max_idx)

    return peaks


def find_spikes(data):
    """
    Finds indexes of local maximums exceeding the impulse threshold.

    :param data: (ndarray) Array of signal data
    :return: (list) Indexes of all local maximums
    """
    spikes_idx = []
    single_spike_idx = []

    # get all indexes at which signal exceeds the impulse threshold
    above_threshold_idx = np.where(data > SPIKE_THRESHOLD)[0]

    for i in range(1, len(above_threshold_idx)):
        # indexes separated at most by MAX_SPIKE_INCONSISTENCY
        # are counted as belonging to the same impulse
        if above_threshold_idx[i] - above_threshold_idx[i - 1] <= MAX_SPIKE_INCONSISTENCY:
            single_spike_idx.append(above_threshold_idx[i])

        # in other case we detect the end of the impulse
        else:
            spikes_idx.append(single_spike_idx)
            single_spike_idx = [above_threshold_idx[i]]

        # include the last impulse
        if i == len(above_threshold_idx) - 1:
            spikes_idx.append(single_spike_idx)

    peaks = _find_peak_indexes(data, spikes_idx)
    return peaks


def calculate_distances(indexes):
    """
    Calculates distances between provided indexes

    :param indexes: (list) List Of indexes
    :return: (list) List of distances between indexes
    """
    distances = []
    for i in range(1, len(indexes)):
        distances.append(indexes[i] - indexes[i - 1])
    return distances
