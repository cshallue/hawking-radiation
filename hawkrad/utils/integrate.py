import numpy as np
from scipy.integrate import quad, solve_ivp


class PiecewiseIntegrator:

  def __init__(self, method, epsabs, epsrel):
    self.method = method
    self.epsabs = epsabs
    self.epsrel = epsrel

    self.value = 0
    self.msgs = []

  def _integrate_quad(self, integrand, x0, x1):
    res = quad(integrand,
               x0,
               x1,
               full_output=True,
               limit=1000,
               epsabs=self.epsabs,
               epsrel=self.epsrel,
               complex_func=True)
    # info_dict has keys 'real' and 'imag'. Error messages are returned as the
    # second element of a tuple.
    msgs = [info[1] for info in res[2].values() if len(info) > 1]
    return res[0], msgs

  def _integrate_ivp(self, integrand, x0, x1):
    sol = solve_ivp(lambda x, y: integrand(x), (x0, x1),
                    y0=np.complex128((0, )),
                    method='DOP853',
                    first_step=1e-8,
                    atol=self.epsabs,
                    rtol=self.epsrel)
    if sol.success:
      return sol.y[0, -1], []

    return 0, [sol.message]

  def __call__(self, integrand, x0, x1):
    if self.method == "quad":
      # There's a bug in scipy.optimize.quad for complex_func=True where it
      # returns the wrong sign when x0 > x1. So ensure that x0 < x1.
      flip, x0, x1 = x1 < x0, min(x0, x1), max(x0, x1)
      res, msgs = self._integrate_quad(integrand, x0, x1)
      if flip:
        res *= -1
    elif self.method == "ivp":
      res, msgs = self._integrate_ivp(integrand, x0, x1)
    else:
      raise ValueError(self.method)

    self.value += res
    self.msgs.extend(msgs)
