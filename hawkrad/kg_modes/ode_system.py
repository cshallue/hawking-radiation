import numpy as np
import scipy.integrate as integrate

from hawkrad.kg_modes.wrappers import ComplexComb, PiecewiseFn, ScaledFn
from hawkrad.schwarz_coords import calc_dxdrstar, calc_rstar

RTOL_MINIMUM = 100 * np.finfo(float).eps


class OdeFailedException(Exception):
  pass


def calc_potential(x, ell):
  lam = ell * (ell + 1)
  x_plus_1 = x + 1
  return x * (lam * x_plus_1 + 1) * x_plus_1**-4


def calc_d2rhodrstar2(x, rho, omega, ell):
  v = calc_potential(x, ell)
  ret = (v - omega**2) * rho
  return ret


# y = (rho, drhodrstar, x)
def calc_dydrstar(rstar, y, omega, ell):
  rho, drhodrstar, x = y
  out = np.zeros_like(y)
  out[0] = drhodrstar
  out[1] = calc_d2rhodrstar2(x, rho, omega, ell)
  out[2] = calc_dxdrstar(x)
  return out


def calc_wronskian(y1, dy1dx, y2, dy2dx):
  return y1 * dy2dx - y2 * dy1dx


def solve_AB(omega, rhoIn, drhoIndrstar, rhoUp, drhoUpdrstar):
  W_in_up = calc_wronskian(rhoIn, drhoIndrstar, rhoUp, drhoUpdrstar)
  W_in_upconj = calc_wronskian(rhoIn, drhoIndrstar, np.conj(rhoUp),
                               np.conj(drhoUpdrstar))
  A_in = -W_in_upconj / W_in_up
  A_up = -np.conj(W_in_upconj) / W_in_up
  B_in = 2j * omega / W_in_up
  B_up = B_in
  return np.array([A_in, A_up, B_in, B_up])


def calc_XY(A_in, A_up, B_in, B_up):
  X_in = 1 / B_in
  X_up = A_up / B_up
  Y_in = A_in / B_in
  Y_up = 1 / B_up
  return np.array([X_in, X_up, Y_in, Y_up])


def solve_ivp(omega,
              ell,
              rho_asym,
              x1,
              x2,
              method='DOP853',
              x_eval=None,
              dense_output=True,
              rtol=None,
              raise_on_failure=True):
  rtol = rtol or RTOL_MINIMUM
  y0 = [rho_asym(x1), rho_asym.deriv_rstar(x1), x1]
  if not np.all(np.isfinite(y0)):
    if raise_on_failure:
      raise OdeFailedException(f"Initial conditions are infinite: {y0}")
    return None

  sol = integrate.solve_ivp(
      calc_dydrstar,
      calc_rstar(np.array([x1, x2])),
      y0=y0,
      method=method,
      t_eval=None if x_eval is None else calc_rstar(np.array(x_eval)),
      dense_output=dense_output,
      args=(omega, ell),
      atol=0,
      rtol=rtol)

  if not sol.success and raise_on_failure:
    raise OdeFailedException(sol.message)
  return sol


# x1 = point closer to horizon (x = 0)
# x2 = point closer to singularity (x = -1)
# Solving proceeds from x1 to x2. Should have x1 > x2.
def solve_ivp_interior(omega, ell, rhoInHoriz, x2, x1=None, **kwargs):
  rhoIn_boundary = -rhoInHoriz.boundary(h_abs_max=5)
  if x1 is None:
    x1 = rhoIn_boundary
  elif x1 > rhoIn_boundary:
    raise ValueError(x1, rhoIn_boundary)

  if (x1 > 0) or (x2 < -1):
    raise ValueError(x1, x2)

  if x1 <= x2:
    # rhoInHoriz is valid from the horizon past x2, so there's nothing to do.
    return None

  return solve_ivp(omega, ell, rhoInHoriz, x1, x2, **kwargs)


# x1 = point closer to horizon (x = 0)
# x2 = point further from horizon (x >> 0)
# Solving proceeds from x1 to x2 for the in mode and x2 to x1 for the up mode.
def solve_ivps_exterior(omega,
                        ell,
                        rhoInHoriz,
                        rhoUpInf,
                        x1=None,
                        x2=None,
                        require_rho_up=False,
                        **kwargs):
  rhoIn_boundary = rhoInHoriz.boundary(h_abs_max=5)
  if x1 is None:
    x1 = rhoIn_boundary
  elif x1 > rhoIn_boundary:
    raise ValueError(x1, rhoIn_boundary)

  rhoUp_boundary = rhoUpInf.boundary()
  if x2 is None:
    # The extra factor of 2 gives us some leeway in the series expansion by
    # making the corrections to the asymptotic form small.
    x2 = 2 * rhoUp_boundary
  elif x2 < rhoUp_boundary:
    raise ValueError(x2, rhoUp_boundary)

  rhoInSol = solve_ivp(omega, ell, rhoInHoriz, x1, x2, **kwargs)

  # If rhoIn gets large, we need to solve for rhoUp numerically because we
  # lose numerical precision by calculating it from rhoIn.
  rhoUpSol = None
  if require_rho_up or (np.max(np.abs(rhoInSol.y[0])) > 10):
    rhoUpSol = solve_ivp(omega, ell, rhoUpInf, x2, x1, **kwargs)

  return rhoInSol, rhoUpSol


def process_ode_sol(rhoSol):
  x1, x2 = rhoSol.y[2, [0, -1]].real
  rhoFn = lambda x: rhoSol.sol(calc_rstar(x))[0]
  return rhoFn, (x1, x2)


def calc_AB_horizon(omega, rhoInHoriz, rhoUpSol):
  rhoUp, drhoUpdrstar, x = rhoUpSol.y[:, -1]
  x = x.real
  rhoIn = rhoInHoriz(x)
  drhoIndrstar = rhoInHoriz.deriv_rstar(x)
  return solve_AB(omega, rhoIn, drhoIndrstar, rhoUp, drhoUpdrstar)


def calc_AB_inf(omega, rhoUpInf, rhoInSol):
  rhoIn, drhoIndrstar, x = rhoInSol.y[:, -1]
  x = x.real
  rhoUp = rhoUpInf(x)
  drhoUpdrstar = rhoUpInf.deriv_rstar(x)
  return solve_AB(omega, rhoIn, drhoIndrstar, rhoUp, drhoUpdrstar)


def make_xi_fns(AB_coeffs,
                rhoInHoriz,
                rhoUpInf,
                rhoInSolExt,
                rhoUpSolExt=None,
                rhoInSolInt=None):
  A_in, A_up, B_in, B_up = AB_coeffs

  # Unpack rhoIn in the exterior, which is required.
  rhoInNumExt, (x1Ext, x2Ext) = process_ode_sol(rhoInSolExt)

  # Make xiIn. Start in the interior, if appplicable.
  xiIn_pieces = []
  if rhoInSolInt is not None:
    rhoInNumInt, (x1Int, x2Int) = process_ode_sol(rhoInSolInt)
    xiIn_pieces.append((ScaledFn(B_in, rhoInNumInt), (x2Int, x1Int)))
  else:
    x1Int = -x1Ext  # rhoInHoriz is valid on both sides of the horizon.

  # xiIn in the exterior.
  xiIn_pieces.append((ScaledFn(B_in, rhoInHoriz), (x1Int, x1Ext)))
  xiIn_pieces.append((ScaledFn(B_in, rhoInNumExt), (x1Ext, x2Ext)))
  xiIn_pieces.append((ComplexComb(A_in, 1, rhoUpInf), (x2Ext, np.inf)))

  # Turn to xiUp, which is defined only in the exterior.
  xiUp_pieces = []
  # Near the horizon.
  xiUp_pieces.append((ComplexComb(A_up, 1, rhoInHoriz), (0, x1Ext)))
  # Numerical region.
  if rhoUpSolExt is not None:
    rhoUpNumExt, (x2ExtUp, x1ExtUp) = process_ode_sol(rhoUpSolExt)
    # rhoUpSol should be valid in the same interval as rhoInSol.
    assert np.allclose(x1Ext, x1ExtUp, rtol=1e-8, atol=1e-8)
    assert np.allclose(x2Ext, x2ExtUp, rtol=1e-8, atol=1e-8)
    xiUpNum = ScaledFn(B_up, rhoUpNumExt)
  else:
    xiUpNum = ComplexComb(A_up, 1, rhoInNumExt)
  xiUp_pieces.append((xiUpNum, (x1Ext, x2Ext)))
  xiUp_pieces.append((ScaledFn(B_up, rhoUpInf), (x2Ext, np.inf)))

  return PiecewiseFn(*xiIn_pieces), PiecewiseFn(*xiUp_pieces)


def solve_modes(omega, ell, rhoInHoriz, rhoUpInf, rmin=1, **kwargs):
  if rmin < 0:
    raise ValueError(rmin)

  xmin = rmin - 1

  rhoInSolInt = None
  if rmin < 1:
    rhoInSolInt = solve_ivp_interior(omega, ell, rhoInHoriz, x2=xmin, **kwargs)

  rhoInSolExt, rhoUpSolExt = solve_ivps_exterior(omega, ell, rhoInHoriz,
                                                 rhoUpInf, **kwargs)

  AB_coeffs = calc_AB_inf(omega, rhoUpInf, rhoInSolExt)
  xiIn, xiUp = make_xi_fns(AB_coeffs, rhoInHoriz, rhoUpInf, rhoInSolExt,
                           rhoUpSolExt, rhoInSolInt)
  return xiIn, xiUp, AB_coeffs
