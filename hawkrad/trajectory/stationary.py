import numpy as np

from hawkrad.schwarz_coords import calc_u_v_region_I_II


def calc_t(tau, R):
  return tau / np.sqrt(1 - 1 / R)


class StationaryTrajectory:

  def __init__(self, R):
    if R <= 1:
      raise ValueError("Stationary observer must be outside the horizon")

    self.R = R

  def calc_r(self, tau):
    return self.R

  def calc_r_u_v(self, tau):
    r = self.R
    t = calc_t(tau, r)
    u, v = calc_u_v_region_I_II(t, r)
    return r, u, v
