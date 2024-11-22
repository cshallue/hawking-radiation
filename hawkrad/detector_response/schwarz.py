import numpy as np

from hawkrad import schwarz_coords
from hawkrad.kg_modes.wrappers import PhiFn, TildeFn
from hawkrad.trajectory.radial_Rinf import RadialTrajectory
from hawkrad.utils.integrate import PiecewiseIntegrator


def make_dtau_integrand(Omega, omega, xiIn, xiUp, AB_coeffs):
  I_in = make_I_in(omega, xiIn)
  I_up = make_I_up(omega, xiIn, xiUp, AB_coeffs)

  def integrand(tau, traj, chi_fn, modename, tilde):
    r, u, v = traj.calc_r_u_v(tau)
    multiplier = chi_fn(tau) * np.exp(-1j * Omega * tau)
    if modename == "in":
      return multiplier * I_in(r, v, tilde)
    if modename == "up":
      return multiplier * I_up(r, u, v, tilde)
    raise ValueError(modename)

  return integrand


def make_I_in(omega, xiIn):
  tildePhiIn = TildeFn(omega, 1, PhiFn(xiIn))

  def I_in(r, v, tilde):
    x = r - 1
    val = tildePhiIn(x) * np.exp(-1j * omega * v)
    if tilde:
      val = np.exp(-2 * np.pi * omega) * np.conj(val)
    return val

  return I_in


def make_I_up(omega, xiIn, xiUp, AB_coeffs):
  _, A_up, B_in, _ = AB_coeffs
  tildePhiIn = TildeFn(omega, 1, PhiFn(xiIn))
  tildePhiUp = TildeFn(omega, -1, PhiFn(xiUp))

  def I_up(r, u, v, tilde):
    x = r - 1

    # Outside the horizon.
    if r > 1:
      val = tildePhiUp(x) * np.exp(-1j * omega * u)
      if tilde:
        val = np.exp(-2 * np.pi * omega) * np.conj(val)
      return val

    # Inside the horizon.
    tildePhi = tildePhiIn(x)
    a = tildePhi * (A_up / B_in) * np.exp(-1j * omega * v)
    b = tildePhi / B_in * np.exp(-1j * omega * u)
    if tilde:
      val = np.exp(-2 * np.pi * omega) * np.conj(a) + b
    else:
      val = a + np.exp(-2 * np.pi * omega) * np.conj(b)
    return val

  return I_up


class TrajectoryIntegrator:

  def __init__(self, method, up_indepvar, delta_tau_horiz, epsabs, epsrel):
    self.method = method
    self.up_indepvar = up_indepvar
    self.delta_tau_horiz = delta_tau_horiz
    self.epsabs = epsabs
    self.epsrel = epsrel

  def _make_integrator(self):
    return PiecewiseIntegrator(self.method, self.epsabs, self.epsrel)

  def __call__(self, integrand, traj, chi_fn, modename, tilde):
    integrator = self._make_integrator()
    regions = [(chi_fn.tau0, chi_fn.tau1)]
    indepvar = "tau"
    r_H = 1

    if modename == "up":
      # If the up integral crosses the horizon, split it into two regions
      # because the integrand is singluar at the horizon.
      r0 = traj.calc_r(chi_fn.tau0)
      r1 = traj.calc_r(chi_fn.tau1)
      crosses_horizon = (r0 - r_H) * (r1 - r_H) < 0
      if crosses_horizon:
        tau_horiz = traj.calc_tau(r_H)
        tau0_horiz = tau_horiz - self.delta_tau_horiz
        tau1_horiz = tau_horiz + self.delta_tau_horiz
        regions = [(chi_fn.tau0, tau0_horiz), (tau1_horiz, chi_fn.tau1)]
      if (self.up_indepvar == "rstar"
          or (crosses_horizon and self.up_indepvar == "rstar_horiz")):
        indepvar = "rstar"
      elif self.up_indepvar != "tau":
        raise ValueError(self.up_indepvar)

    if indepvar == "rstar" and not isinstance(traj, RadialTrajectory):
      # We have tau(r) for the zero mechanical energy case, but not for the
      # general case, so we would need to set up an interpolator or solve
      # the trajectory ODE with rstar as the independent variable.
      raise ValueError("Integrating with respect to rstar is only allowed "
                       "for zero mechanical energy trajectories.")

    # Integrate the regions.
    for tau0p, tau1p in regions:
      if indepvar == "tau":
        integrator(lambda tau: integrand(tau, traj, chi_fn, modename, tilde),
                   tau0p, tau1p)
      elif indepvar == "rstar":
        r0p = traj.calc_r(tau0p)
        r1p = traj.calc_r(tau1p)
        assert (r0p > r_H) == (r1p > r_H)
        region = "I" if r0p > r_H else "II"

        def _integrand(rstar):
          x = schwarz_coords.calc_x(rstar, region)
          r = x + 1
          tau = traj.calc_tau(r)
          dr_dtau = traj.eta * np.sqrt(1 / r)
          dtau_drstar = schwarz_coords.calc_dxdrstar(x) / dr_dtau
          return dtau_drstar * integrand(tau, traj, chi_fn, modename, tilde)

        rstar0p = schwarz_coords.calc_rstar(r0p - 1)
        rstar1p = schwarz_coords.calc_rstar(r1p - 1)
        integrator(_integrand, rstar0p, rstar1p)
      else:
        raise ValueError(indepvar)

    return integrator.value, integrator.msgs
