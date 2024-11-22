import collections
import os
from traceback import format_exception

import numpy as np
from absl import app, flags, logging
from ml_collections import config_dict
from ml_collections.config_flags import config_flags
from scipy.integrate import trapezoid

from hawkrad.detector_response.schwarz import (TrajectoryIntegrator,
                                               make_dtau_integrand)
from hawkrad.kg_modes.asymptotic import (CoeffLoader, load_asymptotic_solution,
                                         load_omegas)
from hawkrad.kg_modes.ode_system import OdeFailedException, solve_modes
from hawkrad.switching import ChiFn
from hawkrad.trajectory import infalling, radial_Rinf, stationary
from hawkrad.utils import Timer, parse_flag_arr

_DEFAULT_CONFIG = config_dict.create(trajectory="infalling",
                                     ode_rtol=1e-10,
                                     int_dtau_method="quad",
                                     int_dtau_up_indepvar="tau",
                                     int_dtau_epsabs=1e-14,
                                     int_dtau_epsrel=1e-8,
                                     delta_tau_horiz=1e-6,
                                     omega_conv_delta=10.0,
                                     omega_epsilon=1e-6)
_DEFAULT_CONFIG.lock()  # Prevent new fields from being added.

config_flags.DEFINE_config_dict("config", _DEFAULT_CONFIG)

flags.DEFINE_string(
    "asym_coeffs_dir", None,
    "Directory containing coefficients of asymptotic solutions.")

flags.DEFINE_string("asym_up_def_file", None,
                    "File containing definition of the up mode coefficients.")

flags.DEFINE_integer(
    "n_asym_coeffs", None,
    "Maximum number of coefficients for the asymptotic solution Taylor series."
)

flags.DEFINE_list(
    "omega", None,
    "Which omegas to evaluate. If not specified, all omegas in the asymptotic "
    "coefficients files will be evaluated.")

flags.DEFINE_float("omega_min", None,
                   "Minimum omega when --omega is not passed.")

flags.DEFINE_float("omega_max", None,
                   "Maximum omega when --omega is not passed.")

flags.DEFINE_list("Omega", "5", "Which detector energy gaps to evaluate.")

flags.DEFINE_list("ell", None, "Which ells to evaluate.", required=True)

flags.DEFINE_list("tau_mid", None, "Values of tau_mid used in the calculation")

flags.DEFINE_list(
    "delta_tau", "0.3",
    "Values of delta_tau. A single value or a list the same length as tau_mid."
)

flags.DEFINE_list("r_mid", None, "Values of r_mid used in the calculation")

flags.DEFINE_list("R_rest", None,
                  "Value or array of radii at which the detector is at rest.")

flags.DEFINE_list("exclude_terms", None,
                  "Terms to exclude, in the form 'vname_modename_tilde'")

flags.DEFINE_bool("write_I_terms", False, "Whether to write the I_terms.")

flags.DEFINE_string("output_dir",
                    None,
                    "Directory in which to save the output.",
                    required=True)

flags.DEFINE_bool("overwrite", False,
                  "Whether to overwrite files in the existing --output_dir.")

FLAGS = flags.FLAGS

# (modename, tilde)
INTEGRANDS = [
    ("in", False),
    ("up", False),
    ("in", True),
    ("up", True),
]

# (vacuum name, modename, tilde)
TERMS = [
    ("B", "in", False),
    ("B", "up", False),
    ("H", "in", False),
    ("H", "up", False),
    ("H", "in", True),
    ("H", "up", True),
]


# We want to explicitly deal with all floating point errors except underflow.
# Underflows may happen naturally, for example in the up integrand inside
# the horizon.
@np.errstate(divide="raise", over="raise", under="ignore", invalid="raise")
def _calculate_terms(Omega_arr, ell, omega, trajectories, rmin, mask, ode_rtol,
                     coeff_loader, integrator):
  timings = {}
  errors = set()

  # Array corresponding to INTEGRANDS, intermediate values used for terms.
  I_terms = np.zeros((len(Omega_arr), len(trajectories), len(INTEGRANDS)))
  # Array corresponding to TERMS, used for the different detector responses.
  terms = np.zeros((len(Omega_arr), len(trajectories), len(TERMS)))
  if terms.shape != mask.shape:
    raise ValueError(terms.shape, mask.shape)

  # Load asymptotic solutions for boundary conditions.
  with Timer() as load_asym_sols_timer:
    try:
      rhoInHoriz = load_asymptotic_solution(coeff_loader, "in", ell, omega)
      rhoUpInf = load_asymptotic_solution(coeff_loader, "up", ell, omega)
    except FloatingPointError as e:
      timings["load_asym_sols"] = load_asym_sols_timer.elapsed
      errors.add("".join(format_exception(e, limit=3)))
      return I_terms, terms, timings, errors

  timings["load_asym_sols"] = load_asym_sols_timer.elapsed

  # Solve for tildePhiIn, tildePhiUp.
  with Timer() as ode_timer:
    try:
      xiIn, xiUp, AB_coeffs = solve_modes(omega,
                                          ell,
                                          rhoInHoriz,
                                          rhoUpInf,
                                          rmin,
                                          rtol=ode_rtol)
    except (FloatingPointError, OdeFailedException) as e:
      timings["ode"] = ode_timer.elapsed
      errors.add("".join(format_exception(e, limit=3)))
      return I_terms, terms, timings, errors

  timings["ode"] = ode_timer.elapsed

  I_computed = np.zeros_like(I_terms, dtype=bool)  # Whether already computed.
  with Timer() as int_timer:
    for i, Omega in enumerate(Omega_arr):
      integrand = make_dtau_integrand(Omega, omega, xiIn, xiUp, AB_coeffs)
      for j, (traj, chi_fn) in enumerate(trajectories):
        for k, (_, modename, tilde) in enumerate(TERMS):
          if not mask[i, j, k]:
            continue
          I_k = INTEGRANDS.index((modename, tilde))
          if not I_computed[i, j, I_k]:
            msgs = []
            try:
              val, msgs = integrator(integrand, traj, chi_fn, modename, tilde)
              term = np.abs(val)**2
            except FloatingPointError as e:
              term = 0.0
              msgs.append("".join(format_exception(e, limit=3)))
            for msg in msgs:
              errors.add(msg)
            I_terms[i, j, I_k] = term
            I_computed[i, j, I_k] = True
          terms[i, j, k] = I_terms[i, j, I_k]

  timings["int_dtau"] = int_timer.elapsed

  for k, (vname, modename, tilde) in enumerate(TERMS):
    try:
      coeff = (2 * ell + 1) / (16 * np.pi**2 * omega)
      if vname == "H":
        x = np.exp(-4 * np.pi * omega)
        if tilde:
          coeff /= (1 - x)
        else:
          # Compute Delta F_H L= F_H - F_B. This converges much faster than
          # F_H because it decays exponentially with omega. In some
          # circumstances we may have already computed F_B (e.g. on a reverse
          # trajectory, since F_B is invariant under reversing the trajectory
          # but F_H is not).
          coeff *= x / (1 - x)
    except FloatingPointError as e:
      coeff = 0
      errors.add("".join(format_exception(e, limit=3)))
    try:
      terms[:, :, k] *= coeff
    except FloatingPointError as e:
      terms[:, :, k] = 0
      errors.add("".join(format_exception(e, limit=3)))

  return I_terms, terms, timings, errors


def _make_trajectories(trajectory):
  # Parse r_mid / tau_mid flags.
  r_mid_arr = parse_flag_arr(FLAGS.r_mid)
  tau_mid_arr = parse_flag_arr(FLAGS.tau_mid)
  if (tau_mid_arr is None) == (r_mid_arr is None):
    raise ValueError("Exactly one of --tau_mid and --r_mid is required")
  num_traj = len(r_mid_arr) if r_mid_arr is not None else len(tau_mid_arr)

  R_rest_arr = parse_flag_arr(FLAGS.R_rest, length=num_traj)
  delta_tau_arr = parse_flag_arr(FLAGS.delta_tau, length=num_traj)
  _write_array("delta_tau", delta_tau_arr)

  if trajectory == "infalling" or trajectory == "outgoing":
    if not FLAGS.R_rest:
      raise ValueError("--R_rest is required")

    if trajectory == "outgoing" and FLAGS.R_rest != ["inf"]:
      raise NotImplementedError(
          "Outgoing trajectories only implemented for R=inf")

    trajectories = []
    for R in R_rest_arr:
      if np.isinf(R):
        trajectories.append(radial_Rinf.RadialTrajectory(trajectory))
      else:
        trajectories.append(infalling.solve_trajectory(R=R))

    if r_mid_arr is None:
      r_mid_arr = np.array([
          traj.calc_r(tau_mid)
          for traj, tau_mid in zip(trajectories, tau_mid_arr)
      ])
    if tau_mid_arr is None:
      tau_mid_arr = np.array([
          traj.calc_tau(r_mid) for traj, r_mid in zip(trajectories, r_mid_arr)
      ])

    _write_array("R_rest", R_rest_arr)
    _write_array("tau_mid", tau_mid_arr)
    _write_array("r_mid", r_mid_arr)

    chi_fns = [
        ChiFn(tau_mid=tau_mid, delta=delta_tau)
        for tau_mid, delta_tau in zip(tau_mid_arr, delta_tau_arr)
    ]
    return list(zip(trajectories, chi_fns))
  elif trajectory == "stationary":
    # Stationary trajectories pass R via r_mid.
    if r_mid_arr is None:
      raise ValueError("--r_mid is required for stationary trajectory")
    if R_rest_arr is not None:
      raise ValueError("--R_rest is disallowed for stationary trajectory")
    R_arr = r_mid_arr
    _write_array("R", R_arr)

    # Just set tau_mid = 0 for all trajectories.
    return [(stationary.StationaryTrajectory(R),
             ChiFn(tau_mid=0, delta=delta_tau))
            for R, delta_tau in zip(R_arr, delta_tau_arr)]
  else:
    raise ValueError(trajectory)


def _update_mask(i, omega_arr, terms, mask, delta_omega, epsilon):
  omega = omega_arr[i]
  # Compare to the largest omega at least delta smaller than the current.
  idxs = np.where(omega_arr < omega - delta_omega)[0]
  if not len(idxs):
    return

  j = idxs[-1]
  old_omega = omega_arr[j]
  old_int_terms = trapezoid(terms[:j + 1], omega_arr[:j + 1], axis=0)
  new_int_terms = trapezoid(terms[:i + 1], omega_arr[:i + 1], axis=0)
  # Normalize the interval to delta_omega length.
  diff = (new_int_terms - old_int_terms) * delta_omega / (omega - old_omega)
  # Has the integral converged?
  old_n_converged = np.sum(~mask)
  mask &= (diff > epsilon * new_int_terms)
  n_converged = np.sum(~mask)
  if n_converged > old_n_converged:
    logging.info(f"{n_converged} out of {mask.size} integrals converged at "
                 f"omega = {omega:.3g}")


def _write_array(name, arr, verbose=True):
  np.save(os.path.join(FLAGS.output_dir, f"{name}.npy"), arr)
  if verbose:
    logging.info(f"{name} = {arr}")


def main(unused_argv):
  FLAGS.alsologtostderr = True
  config = FLAGS.config

  if os.path.exists(FLAGS.output_dir):
    if not FLAGS.overwrite:
      raise ValueError("--output_dir already exists and --overwrite is False")

  # Stream logs to stderr and to disk.
  log_dir = os.path.join(FLAGS.output_dir, "logs")
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  logging.get_absl_handler().use_absl_log_file(log_dir=log_dir)
  FLAGS.alsologtostderr = True

  # Save the config.
  config_json = config.to_json(indent=2)
  logging.info(f"Config:\n{config_json}")
  with open(os.path.join(FLAGS.output_dir, "config.json"), "w") as f:
    f.write(config_json)

  # Load the Omega, ell, omega arrays.
  Omega_arr = parse_flag_arr(FLAGS.Omega)
  _write_array("energy_gap", Omega_arr)

  ell_arr = parse_flag_arr(FLAGS.ell, dtype=int)
  _write_array("ell", ell_arr)

  if FLAGS.omega is not None:
    if FLAGS.omega_min is not None or FLAGS.omega_max is not None:
      raise ValueError("--omega cannot be used in conjunction with "
                       "--omega_min or --omega_max")
    omega_arr = parse_flag_arr(FLAGS.omega)
  else:
    if FLAGS.asym_coeffs_dir is None:
      raise ValueError("One of --omega and --asym_coeffs_dir is required")
    omega_arr = load_omegas(FLAGS.asym_coeffs_dir,
                            omega_min=FLAGS.omega_min,
                            omega_max=FLAGS.omega_max)
  _write_array("omega", omega_arr)

  # Make the trajectories.
  trajectories = _make_trajectories(config.trajectory)
  rmin_arr = np.array(
      [traj.calc_r(chi_fn.tau1) for traj, chi_fn in trajectories])
  rmin = np.min(rmin_arr)
  rmin -= 1e-3  # Solve to slightly closer to r=0 to give us a bit of room.
  logging.info(f"r min: {rmin:.3g}")
  if rmin < 0:
    raise ValueError(rmin)

  coeff_loader = CoeffLoader(FLAGS.asym_coeffs_dir, FLAGS.asym_up_def_file,
                             FLAGS.n_asym_coeffs)
  integrator = TrajectoryIntegrator(method=config.int_dtau_method,
                                    up_indepvar=config.int_dtau_up_indepvar,
                                    delta_tau_horiz=config.delta_tau_horiz,
                                    epsabs=config.int_dtau_epsabs,
                                    epsrel=config.int_dtau_epsrel)
  omega_eps = config.omega_epsilon
  for ell in ell_arr:
    logging.info(f"ell = {ell}")
    ell_timer = Timer()
    ell_timer.start()
    timings = collections.Counter()
    errors = collections.defaultdict(list)

    I_terms = np.zeros(
        (len(omega_arr), len(Omega_arr), len(trajectories), len(INTEGRANDS)))
    terms = np.zeros(
        (len(omega_arr), len(Omega_arr), len(trajectories), len(TERMS)))

    # The mask array indicates tau_mid's and in/up modes for which we need to
    # keep computing omegas.
    mask = np.ones_like(terms[0], dtype=bool)
    # Explicity excluded terms.
    for exclude in FLAGS.exclude_terms or []:
      e = exclude.split("_")
      excluded = False
      for k, term in enumerate(TERMS):
        if (term[0] == e[0]) and (term[1] == e[1]) and (str(term[2]) == e[2]):
          logging.info(f"Excluding term {term}")
          mask[:, :, k] = False
          excluded = True
          break
      if not excluded:
        raise ValueError(f"No term to exclude matching {e}!")

    for i, omega in enumerate(omega_arr):
      logging.log_every_n_seconds(
          logging.INFO, f"omega = {omega:.4g} ({i+1} out of {len(omega_arr)})",
          30)
      I_terms[i], terms[i], ts, es = _calculate_terms(Omega_arr, ell, omega,
                                                      trajectories, rmin, mask,
                                                      config.ode_rtol,
                                                      coeff_loader, integrator)
      timings.update(ts)
      for msg in es:
        errors[msg].append(omega)

      # Check for convergence.
      if omega_eps > 0 and omega > config.omega_conv_delta:
        _update_mask(i, omega_arr, terms, mask, config.omega_conv_delta,
                     config.omega_epsilon)

      # Break out of omega loop if all components have converged.
      if not np.any(mask):
        logging.info(f"All omega integrals converged at omega = {omega:.2f}")
        break

    # Finished this ell.
    if FLAGS.write_I_terms:
      _write_array(f"I_terms_ell{ell}", I_terms, verbose=False)
    _write_array(f"terms_ell{ell}", terms, verbose=False)
    ell_timer.stop()
    timings["total"] = ell_timer.elapsed
    logging.info("Timings: " +
                 ", ".join([f"{k}: {t:.2f} s" for k, t in timings.items()]))
    for msg, omegas in errors.items():
      logging.warning(f"Warning for omega = {np.array(omegas)}:\n{msg}")


if __name__ == "__main__":
  app.run(main)
