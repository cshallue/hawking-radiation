import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize

from hawkrad.schwarz_coords import (calc_kappa_region_I_II, calc_rstar,
                                    calc_u_v_region_I_II)
from hawkrad.utils import apply_to_array


# E = conserved energy. R = radius of zero velocity.
def calc_E(R):
  return np.sqrt(1 - 1 / R)


def calc_R(E):
  return 1 / (1 - E**2)


# Geodesic equations with tau as the parameter.
# Valid in regions I and II.


def calc_dr_dtau(r, E):
  dr_dtau_sq = 1 / r + E**2 - 1
  if np.any(dr_dtau_sq < 0):
    raise ValueError(dr_dtau_sq)
  return -np.sqrt(dr_dtau_sq)


def calc_dV_dtau_over_V(r, E):
  return (E + calc_dr_dtau(r, E)) / (1 - 1 / r) / 2


def calc_dU_dtau_over_U(r, E):
  return -(E - calc_dr_dtau(r, E)) / (1 - 1 / r) / 2


def calc_dv_dtau(r, E):
  # sigma = 1 in both regions I and II
  return 2 * calc_dV_dtau_over_V(r, E)


def calc_du_dtau(r, E):
  k = calc_kappa_region_I_II(r)
  return 2 * k * calc_dU_dtau_over_U(r, E)


class TrajectorySolution:

  def __init__(self, sol):
    self.sol = sol
    self.tau_min = sol.t[0]
    self.tau_max = sol.t[-1]

  def calc_r_v(self, tau, range_err=True):
    mask = (tau >= self.tau_min) & (tau <= self.tau_max)
    if range_err and not np.all(mask):
      raise ValueError(
          f"tau must be in the range [{self.tau_min:.3g}, {self.tau_max:.3g}]")

    return np.where(mask, self.sol.sol(tau), np.nan)

  def calc_r_u_v(self, tau, range_err=True):
    r, v = self.calc_r_v(tau, range_err)
    k = calc_kappa_region_I_II(r)
    rstar = calc_rstar(r - 1)
    u = k * (2 * rstar - v)
    return r, u, v

  def calc_r(self, tau, range_err=True):
    return self.calc_r_v(tau, range_err)[0]

  def calc_v(self, tau, range_err=True):
    return self.calc_r_v(tau, range_err)[1]

  def calc_tau(self, r):
    if np.size(r) > 1:
      return apply_to_array(r, self.calc_tau)
    return optimize.bisect(lambda t: self.calc_r(t) - r, self.tau_min,
                           self.tau_max)


def solve_trajectory(E=None,
                     R=None,
                     r0=None,
                     tau_max=None,
                     rtol=1e-8,
                     raise_on_failure=True):
  # By default, solve from the position at which the trajectory is at rest.
  if r0 is None:
    if R is None or np.isinf(R):
      raise ValueError("r0 is required unless finite R is passed")
    # We can't set r0=R exactly because dr_dtau is zero there, so r will never
    # change because the system is first order.
    r0 = R - 1e-10

  if (E is None) == (R is None):
    raise ValueError("Exactly one of E or R is required")

  if E is None:
    E = calc_E(R)

  if R is None:
    R = calc_R(E)

  if tau_max is None:
    if R <= 1:
      raise ValueError("tau_max cannot be inferred if R < 1")
    # Solve almost to the singularity.
    tau_max = calc_tau(np.pi, R) - 1e-4

  # y = (r, v)
  def calc_dydtau(tau, y):
    r, v = y
    out = np.zeros_like(y)
    out[0] = calc_dr_dtau(r, E)
    out[1] = calc_dv_dtau(r, E)
    return out

  # Set tau=0 and t=0 at the start of the trajectory.
  tau0, t0 = 0, 0
  _, v0 = calc_u_v_region_I_II(t0, r0)
  y0 = (r0, v0)

  sol = integrate.solve_ivp(calc_dydtau, (tau0, tau_max),
                            y0=y0,
                            method='DOP853',
                            dense_output=True,
                            atol=0,
                            rtol=rtol)

  if not sol.success and raise_on_failure:
    raise ValueError(sol.message)

  return TrajectorySolution(sol)


# Geodesic equations with rstar as the parameter.
# Valid in regions I and II.
#
# This system only works for r suitably smaller than R because calc_dtau_drstar
# is infinite at r=R.


def calc_dtau_drstar(r, E):
  return (1 - 1 / r) / calc_dr_dtau(r, E)


def calc_dv_drstar(r, E):
  s = 1  # sigma = 1 in both regions I and II
  return s * (1 + E / calc_dr_dtau(r, E))


def calc_du_drstar(r, E):
  k = calc_kappa_region_I_II(r)
  return k * (1 - E / calc_dr_dtau(r, E))


# Explicit solution in Schwarzschild coordinates in terms of the cycloid
# parameter eta. Valid in regions I and II.
#
# The range of eta is (-pi, pi). For negative eta, the trajectory is radially
# outward, whereas for positive eta it is radially inward.
#
# tau = 0 corresponds to eta = 0, which is when the trajectory turns around.


def calc_tau(eta, R):
  return R**(3 / 2) * (eta + np.sin(eta)) / 2


def calc_r(eta, R):
  return R * (1 + np.cos(eta)) / 2


def calc_t(eta, R):
  return ((R - 1)**(1 / 2) * (eta + R * (eta + np.sin(eta)) / 2) + np.log(
      np.abs((R - 1)**(1 / 2) + np.tan(eta / 2)) /
      np.abs((R - 1)**(1 / 2) - np.tan(eta / 2))))
