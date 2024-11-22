import collections
import os

import numpy as np
from absl import app, flags, logging

from hawkrad.kg_modes.asymptotic import CoeffLoader, load_asymptotic_solution
from hawkrad.kg_modes.ode_system import solve_modes
from hawkrad.kg_modes.wrappers import PhiFn
from hawkrad.utils import Timer, parse_flag_arr

flags.DEFINE_string(
    "asym_coeffs_dir",
    None,
    "Directory containing coefficients of asymptotic solutions",
    required=True)

flags.DEFINE_list("omega", None, "Which omegas to evaluate.", required=True)

flags.DEFINE_integer("ell_max", None, "Maximum ell to evaluate.")

flags.DEFINE_float("ell_epsilon", 1e-6,
                   "Convergence criterion for the ell sums.")

flags.DEFINE_list("R",
                  None,
                  "Radii at which to calculate the transition rate",
                  required=True)

flags.DEFINE_string("output_dir",
                    None,
                    "Directory in which to save the output.",
                    required=True)

flags.DEFINE_bool("overwrite", False,
                  "Whether to overwrite files in the existing --output_dir.")

FLAGS = flags.FLAGS

MODES = ["in", "up"]


def _parse_R_arr():
  if len(FLAGS.R) == 1 and os.path.exists(FLAGS.R[0]):
    R_arr = np.load(FLAGS.R[0])
  else:
    R_arr = np.array([R for R in FLAGS.R if R], dtype=float)

  R_str = ", ".join(f"{R:.3g}" for R in R_arr)
  logging.info(f"R = [{R_str}]")

  return R_arr


def _calculate_terms(coeff_loader, ell, omega, R_arr, mask):
  # Solve for tildePhiIn, tildePhiUp.
  rmin = np.min(np.where(mask, R_arr, np.inf))
  with Timer() as ode_timer:
    rhoInHoriz = load_asymptotic_solution(coeff_loader, "in", ell, omega)
    rhoUpInf = load_asymptotic_solution(coeff_loader, "up", ell, omega)
    xiIn, xiUp, _ = solve_modes(omega, ell, rhoInHoriz, rhoUpInf, rmin)
    phiFns = {"in": PhiFn(xiIn), "up": PhiFn(xiUp)}

  F_arr = np.zeros((len(R_arr), len(MODES)), dtype=float)
  with Timer() as terms_timer:
    for i, R in enumerate(R_arr):
      if not mask[i]:
        continue
      for j, modename in enumerate(MODES):
        x = R - 1
        F_arr[i, j] = np.abs(phiFns[modename](x))**2
    F_arr *= (2 * ell + 1)

  timings = {"ode": ode_timer.elapsed, "calc_terms": terms_timer.elapsed}
  return F_arr, timings


def _write_array(name, arr):
  np.save(os.path.join(FLAGS.output_dir, f"{name}.npy"), arr)


def main(unused_argv):
  FLAGS.alsologtostderr = True

  if os.path.exists(FLAGS.output_dir):
    if not FLAGS.overwrite:
      raise ValueError("--output_dir already exists and --overwrite is False")
  else:
    os.makedirs(FLAGS.output_dir)

  R_arr = _parse_R_arr()
  rmin = np.min(R_arr)
  if rmin < 1:
    raise ValueError(rmin)  # Working outside the horizon only.

  omega_arr = parse_flag_arr(FLAGS.omega)
  ell_arr = np.arange(FLAGS.ell_max + 1, dtype=int)

  # Save omega, ell, and tau.
  _write_array("R", R_arr)
  _write_array("omega", omega_arr)
  _write_array("ell", ell_arr)

  coeff_loader = CoeffLoader(FLAGS.asym_coeffs_dir)

  # Prepare output arrays.
  F_arr = np.zeros((len(ell_arr), len(omega_arr), len(R_arr), len(MODES)))

  # Array used to keep track of convergence.
  # For each ell, the mask array indicates R's for which we need to keep
  # computing omegas.
  mask = np.ones_like(R_arr, dtype=bool)

  ell_eps = FLAGS.ell_epsilon
  for ell in ell_arr:
    logging.info(f"ell = {ell}")
    ell_timer = Timer()
    ell_timer.start()
    timings = collections.Counter()

    for i, omega in enumerate(omega_arr):
      new_Fs, ts = _calculate_terms(coeff_loader, ell, omega, R_arr, mask)
      F_arr[ell, i] = new_Fs
      timings.update(ts)

    # Save the partially filled out array.
    with Timer() as write_timer:
      _write_array("F_in_up", F_arr)
    timings["write_output"] = write_timer.elapsed

    ell_timer.stop()
    timings["total"] = ell_timer.elapsed
    logging.info("Timings: " +
                 ", ".join([f"{k}: {t:.2f} s" for k, t in timings.items()]))

    # Which ell sums have converged? Make the mask for the next ell.
    mask &= np.any((F_arr[ell] > ell_eps * np.sum(F_arr, axis=0)), axis=(0, 2))
    n_converged = mask.size - np.sum(mask)
    logging.info(f"{n_converged:,} out of {mask.size:,} ell sums completed")

    # Break out of the ell loop if all components have converged.
    if not np.any(mask):
      logging.info(f"All ell sums completed at ell = {ell}")
      break


if __name__ == "__main__":
  app.run(main)
