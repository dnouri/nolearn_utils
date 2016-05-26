import numpy as np
from nolearn.lasagne import BatchIterator


def make_iterator(name, mixins):
    """
    Return an Iterator class added with the provided mixins

    Parameters
    ----------
    name : string
        Name of the iterator

    mixins : list of classes
        List of mixin classes to be used.
        The first mixin in the list will be applied first

    Returns
    -------
    TODO
    """
    mixins = [BatchIterator] + mixins
    # Reverse the list for type()
    mixins.reverse()
    return type(name, tuple(mixins), {})


def get_random_idx(arr, p):
    """
    Return indices which some of the elements will be selected according to the
    specified probabilitiy

    Parameters
    ----------
    arr : numpy array

    p : float
        Probability of each elements to be picked
        0 means no index will be returned and 1 means all indices will be
        returned

    Returns
    -------
    integer indices in form of numpy array
    """
    n = arr.shape[0]
    idx = np.random.choice(n, int(n * p), replace=False)
    return idx
