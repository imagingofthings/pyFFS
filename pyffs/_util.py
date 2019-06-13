# #############################################################################
# _util.py
# ========
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################


def _index(x, axis, index_spec):
    """
    Form indexing tuple for NumPy arrays.

    Given an array `x`, generates the indexing tuple that has :py:class:`slice` in each axis except
    `axis`, where `index_spec` is used instead.

    Parameters
    ----------
    x : :py:class:`~numpy.ndarray`
        Array to index.
    axis : int
        Dimension along which to apply `index_spec`.
    index_spec : int or :py:class:`slice`
        Index/slice to use.

    Returns
    -------
    indexer : tuple
        Indexing tuple.
    """
    idx = [slice(None)] * x.ndim
    idx[axis] = index_spec

    indexer = tuple(idx)
    return indexer
