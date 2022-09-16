import numpy as np


def generate_sequences(array, length):
    """
    Generate overlapping sequences of size length by slicing the array
    """
    sequences = []
    # iterate over the array
    for i in range(0, array.shape[0] - length):
        # create subsections of the array by slicing overlapping sections of size length
        sequences.append(array[i:i+length])
    return sequences


class Sequences:
    """
    Assume that a sequence of nth-order derivatives are correlated.
    """
    def __init__(self, array, length):
        self._length = length
        self._sequence_list = generate_sequences(array, length)

    def add_sequences(self, sequences):
        """
            Add a 1 or more sequences to the private class field
        """
        if len(sequences) == self._length:
            # sequence is not iterable
            self._sequence_list.append(sequences)
        else:
            # sequence is iterable
            self._sequence_list.extend(sequences)

    def compare_local(self, sequence, k):
        """
            Compare a sequence of length-1 as a numpy vector to the sequences of length,
            derive the next value in the sequence as the average of the k nearest neighbours (not k-NN method).
        """
        #
        if sequence.shape[0] != self._length-1:
            raise ValueError("sequence length is not 1 less than sequences, sequence length is {} and \
                             sequences length is {}".format(sequence.shape[0], self._length))
        # pad sequence to the correct length
        pad_sequence = np.zeros((self._length,))  # predefine array size
        pad_sequence[0:self._length-1] = sequence  # populate array
        # leverage SIMD processing in numpy arrays
        idx_list = np.arange(0, len(self._sequence_list))
        dist_list = np.sqrt(np.sum(np.square(self._sequence_list - pad_sequence), axis=1))  # calculate distance
        # idx_dist_list = [(i, dist_list) for i in range(dist_list.shape[0])]  # index distance values
        # sort indexed distance list
        sorted_list = sorted(zip(idx_list, dist_list), key=lambda x: x[1], reverse=False)
        # isolate the k nearest neighbours
        nearest_neighbours = sorted_list[0: k]
        # calculate mean and standard div
        running_total = 0
        running_total_square = 0
        for neighbour_idx, neighbour_dist in nearest_neighbours:
            running_total += self._sequence_list[neighbour_idx][-1]
            running_total_square += self._sequence_list[neighbour_idx][-1]**2
        mean = running_total/k
        local_var = running_total_square/k - mean**2
        return mean, local_var
