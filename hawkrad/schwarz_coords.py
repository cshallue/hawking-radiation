import numpy as np
from scipy.special import lambertw, wrightomega

# As everywhere in the code, r means r / 2M and rstar means rstar / 2M.
# x means r - 1. For r within about 1e-16 of the horizon, r is effectively
# truncated to 1.0, whereas using x supports a much wider range of rstar.


def get_sigma(region):
  if region == "I" or region == "II":
    return 1
  if region == "III" or region == "IV":
    return -1
  raise ValueError(region)


def get_kappa(region):
  if region == "II" or region == "III":
    return 1
  if region == "I" or region == "IV":
    return -1
  raise ValueError(region)


def calc_kappa_region_I_II(r):
  return np.where(r <= 1, 1, -1)


def calc_rstar(x):
  return 1 + x + np.log(np.abs(x))


def calc_x(rstar, region="I"):
  sk = get_sigma(region) * get_kappa(region)
  if sk < 0:
    # Use the Wright omega function to avoid overflow in the exponential for
    # large rstar.
    return wrightomega(rstar - 1)

  # Overflow isn't a problem because rstar < 0.
  # Underflow doesn't seem to be a problem.
  return lambertw(-np.exp(rstar - 1)).real


def calc_drstardx(x):
  return 1 + 1 / x


def calc_dxdrstar(x):
  return 1 - 1 / (x + 1)


def calc_t_rstar(u, v, region):
  s = get_sigma(region)
  k = get_kappa(region)
  t = (s * v - k * u) / 2
  rstar = (s * v + k * u) / 2
  return t, rstar


def calc_u_v(t, rstar, region):
  s = get_sigma(region)
  k = get_kappa(region)
  u = k * (rstar - t)
  v = s * (rstar + t)
  return u, v


# This doesn't work near the horizon because both rstar and t go to infinity
# and machine precision in the difference becomes an issue.
def calc_u_v_region_I_II(t, r):
  rstar = calc_rstar(r - 1)
  s = 1
  k = calc_kappa_region_I_II(r)
  u = k * (rstar - t)
  v = s * (rstar + t)
  return u, v
