import os

import numpy as np


def parse_flag_arr(flag_vals, length=None, dtype=float):
  if flag_vals is None:
    return

  if len(flag_vals) == 1 and flag_vals[0].endswith(".npy"):
    return np.load(flag_vals[0]).astype(dtype)

  out = []
  for val in flag_vals:
    if ":" in val:
      out.extend(np.arange(*(dtype(x) for x in val.split(":")), dtype=dtype))
    else:
      out.append(dtype(val))

  if length is not None:
    if len(out) == 1:
      out = out * length
    elif len(out) != length:
      raise ValueError(
          f"Expected a list of length 1 or {length}. Got: {len(out)}")

  return np.array(out, dtype=dtype)
