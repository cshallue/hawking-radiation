import numpy as np


def apply_to_array(arr, fn, *args):
  if np.ndim(arr) != 1:
    # Could easily support higher dimensional arrays, but not needed right now.
    raise ValueError("Only 1d arrays supported")

  res = np.zeros_like(arr)
  for i, x in enumerate(arr):
    res[i] = fn(x, *args)
  return res
