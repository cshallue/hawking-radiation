import numpy as np

from hawkrad.schwarz_coords import calc_rstar


class PiecewiseFn:

  def __init__(self, *fns):
    self.fns = fns

  def __call__(self, x, range_err=True):
    if np.size(x) == 1:
      for fn, (xmin, xmax) in self.fns:
        if (x >= xmin) and (x < xmax):
          return fn(x)
      if range_err:
        raise ValueError(f"x out of range: {x:.4g}")
      return np.nan

    y = np.zeros_like(x, dtype=np.complex128) * np.nan
    is_evaluated = np.zeros_like(x, dtype=bool)
    for fn, (xmin, xmax) in self.fns:
      cond = (x >= xmin) & (x < xmax)
      xvals = x[cond]
      if xvals.size > 0:
        y[cond] = fn(xvals)
        is_evaluated[cond] = True
    if range_err and not np.all(is_evaluated):
      raise ValueError(f"x out of range: {x[~is_evaluated]}")
    return y


class ScaledFn:

  def __init__(self, scale, fn):
    self.scale = scale
    self.fn = fn

  def __call__(self, x):
    return self.scale * self.fn(x)


# A * fn(x) + B * conj(fn(x))
class ComplexComb:

  def __init__(self, A, B, fn):
    self.A = A
    self.B = B
    self.fn = fn

  def __call__(self, x):
    val = self.fn(x)
    return self.A * val + self.B * np.conj(val)


class PhiFn:

  def __init__(self, xi_fn):
    self.xi_fn = xi_fn

  def __call__(self, x):
    return self.xi_fn(x) / (x + 1)


class TildeFn:

  def __init__(self, omega, sgn, fn):
    self.omega = sgn * omega
    self.fn = fn

  def __call__(self, x):
    rstar = calc_rstar(x)
    return np.exp(1j * self.omega * rstar) * self.fn(x)


class TildePhiFn:

  def __init__(self, omega, sgn, piecewiseXiFn, horizon_delta=1e-5):
    self.omega = sgn * omega

    pieces = []
    for xiFn, (xmin, xmax) in piecewiseXiFn.fns:
      tildePhiFn = TildeFn(omega, sgn, PhiFn(xiFn))
      # If we cross the horizon, insert the smooth solution.
      if (xmin < 0) and (xmax > 0):
        delta = horizon_delta
        pieces.append((tildePhiFn, (xmin, -delta)))
        B_in = xiFn.scale
        pieces.append((lambda x: B_in / (x + 1), (-delta, delta)))
        pieces.append((tildePhiFn, (delta, xmax)))
      else:
        pieces.append((tildePhiFn, (xmin, xmax)))

    self.fn = PiecewiseFn(*pieces)

  def __call__(self, x):
    return self.fn(x)
