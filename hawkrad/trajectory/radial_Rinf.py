import numpy as np

from hawkrad.schwarz_coords import calc_kappa_region_I_II

# Explicit solution for a radial geodesic with zero mechanical energy.
# eta = -1 for infalling, +1 for outgoing.
# Valid in regions I and II.


def calc_tau(r, eta):
  return eta * 2 / 3 * r**(3 / 2)


def calc_r(tau, eta):
  return (eta * 3 / 2 * tau)**(2 / 3)


def calc_u(r, eta):
  k = calc_kappa_region_I_II(r)
  sqrt_r = r**(1 / 2)
  ku = r - 2 * eta * (sqrt_r**3 / 3 + sqrt_r) + 2 * np.log(-k * (sqrt_r + eta))
  return k * ku


def calc_v(r, eta):
  sqrt_r = r**(1 / 2)
  return r + 2 * eta * (sqrt_r**3 / 3 + sqrt_r) + 2 * np.log(sqrt_r - eta)


def calc_r_u_v(tau, eta):
  r = calc_r(tau, eta)
  u = calc_u(r, eta)
  v = calc_v(r, eta)
  return r, u, v


def calc_U(r, eta):
  sqrt_r = r**(1 / 2)
  return -(sqrt_r + eta) * np.exp(r / 2 - eta * (sqrt_r**3 / 3 + sqrt_r))


def calc_V(r, eta):
  sqrt_r = r**(1 / 2)
  return (sqrt_r - eta) * np.exp(r / 2 + eta * (sqrt_r**3 / 3 + sqrt_r))


class RadialTrajectory:

  def __init__(self, direction):
    if direction == "infalling":
      self.eta = -1
    elif direction == "outgoing":
      self.eta = 1
    else:
      raise ValueError(direction)

  def calc_tau(self, r):
    return calc_tau(r, self.eta)

  def calc_r(self, tau):
    return calc_r(tau, self.eta)

  def calc_u(self, r):
    return calc_u(r, self.eta)

  def calc_v(self, r):
    calc_v(r, self.eta)

  def calc_r_u_v(self, tau):
    return calc_r_u_v(tau, self.eta)

  def calc_U(self, r):
    return calc_U(r, self.eta)

  def calc_V(self, r):
    return calc_V(r, self.eta)
