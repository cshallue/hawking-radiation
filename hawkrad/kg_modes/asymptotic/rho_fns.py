from abc import ABC, abstractmethod

import numpy as np
from scipy import optimize

from hawkrad.schwarz_coords import calc_drstardx, calc_dxdrstar, calc_rstar


class HFn(ABC):

  @abstractmethod
  def __call__(self, x):
    ...

  @abstractmethod
  def deriv(self, x):
    ...

  @abstractmethod
  def dhdx_over_h(self, x):
    ...


class HFnAsym(HFn):

  def __call__(self, x):
    return 1

  def deriv(self, x):
    return 0

  def dhdx_over_h(self, x):
    return 0


class RhoFn:

  def __init__(self, omega, sgn, h):
    self.omega = sgn * omega
    self.h = h

  def __call__(self, x):
    return np.exp(1j * self.omega * calc_rstar(x)) * self.h(x)

  def deriv_x(self, x):
    return self(x) * (1j * self.omega * calc_drstardx(x) +
                      self.h.dhdx_over_h(x))

  def deriv_rstar(self, x):
    return self(x) * (1j * self.omega +
                      self.h.dhdx_over_h(x) * calc_dxdrstar(x))


class VFn:

  def __init__(self, c):
    self.vu = np.polynomial.Polynomial(c)  # v(u), where u = 1/(x+1)
    self.dvdu = self.vu.deriv()  # dvdu(u)

  @staticmethod
  def u(x):
    return 1 / (x + 1)

  @staticmethod
  def dudx(x):
    return -(x + 1)**-2

  def __call__(self, x):
    return self.vu(self.u(x))

  def deriv(self, x):
    return self.dvdu(self.u(x)) * self.dudx(x)


class HFnUp(HFn):

  def __init__(self, v):
    self.v = v

  def __call__(self, x):
    return np.exp(self.v(x))

  def deriv(self, x):
    return self(x) * self.dhdx_over_h(x)

  def dhdx_over_h(self, x):
    return self.v.deriv(x)


class RhoFnUpInf(RhoFn):

  def __init__(self, omega, c):
    super().__init__(omega, 1, HFnUp(VFn(c)))
    self.c = c

  def boundary(self, accuracy=1e-16):
    n0 = np.nonzero(np.abs(self.c) > 1e-16)[0][0]  # Index of first nonzero c
    ninf = len(self.c) - 1
    ratio = np.abs(self.c[ninf] / self.c[n0])
    # Work with logarithms to avoid underflow.
    return np.exp((np.log(accuracy) - np.log(ratio)) / (n0 - ninf)) - 1


# This is equivalent to the above class with v(x) = 0.
class RhoFnUpInfAsym(RhoFn):

  def __init__(self, omega):
    super().__init__(omega, 1, HFnAsym())


class HFnIn(HFn):

  def __init__(self, b):
    self.w = np.polynomial.Polynomial(b)  # w(x)
    self.dwdx = self.w.deriv()  # dwdx(x)

  def __call__(self, x):
    return self.w(x)

  def deriv(self, x):
    return self.dwdx(x)

  def dhdx_over_h(self, x):
    return self.deriv(x) / self(x)


class RhoFnInHoriz(RhoFn):

  def __init__(self, omega, b):
    super().__init__(omega, -1, HFnIn(b))
    self.b = b

  def boundary(self, accuracy=1e-16, h_abs_max=None):
    ninf = len(self.b) - 1
    # Here we're assuming b[0] = 1.
    # Work with logarithms to avoid underflow.
    x_b = np.exp((np.log(accuracy) - np.log(np.abs(self.b[ninf]))) / ninf)

    # Move the boundary to the left to ensure |h(x_b)| < h_abs_max.
    if h_abs_max and np.abs(self.h(x_b)) > h_abs_max:
      x_b = optimize.bisect(lambda x: np.abs(self.h(x)) - h_abs_max,
                            0,
                            x_b,
                            xtol=1e-8)  # Don't need to be super precise.
    return x_b


# This is equivalent to the above class with w(x) = 1.
class RhoFnInHorizAsym(RhoFn):

  def __init__(self, omega):
    super().__init__(omega, -1, HFnAsym())
