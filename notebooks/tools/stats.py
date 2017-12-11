import numpy as np


def norm2one(x, bins, range):
    """Compute normalization weights for histogram."""
    return np.ones(x.size)/np.histogram(x, bins, range)[0].max()
