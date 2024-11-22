import os

import numpy as np
from absl import app, flags, logging

from hawkrad.kg_modes.asymptotic import CoeffLoader
from hawkrad.utils import parse_flag_arr

flags.DEFINE_string("output_dir", None, "Output directory.", required=True)

flags.DEFINE_string("existing_tables_dir", None,
                    "Directory containing existing coefficients tables.")

flags.DEFINE_string("asym_up_def_file", None,
                    "File containing definition of the up mode coefficients.")

flags.DEFINE_integer(
    "n_asym_coeffs",
    None,
    "Maximum number of coefficients for the asymptotic solution Taylor series.",
    required=True)

flags.DEFINE_list("modenames", None, "Which modenames to generate.")

flags.DEFINE_bool("overwrite", False,
                  "Whether to overwrite files in the existing --output_dir.")

flags.DEFINE_list("ell", None, "Which ells to evaluate.", required=True)

flags.DEFINE_list("omega", None, "Which omegas to evaluate.", required=True)

FLAGS = flags.FLAGS


def main(unused_argv):
  ell_arr = parse_flag_arr(FLAGS.ell, dtype=int)
  omega_arr = parse_flag_arr(FLAGS.omega)
  logging.info(f"omega = {omega_arr}")

  coeff_loader = CoeffLoader(FLAGS.existing_tables_dir, FLAGS.asym_up_def_file,
                             FLAGS.n_asym_coeffs)

  n_max = FLAGS.n_asym_coeffs
  for modename in FLAGS.modenames:
    outdir = os.path.join(FLAGS.output_dir, modename)
    if not os.path.exists(outdir):
      os.makedirs(outdir)

    # Save the omega array, but make sure any existing file matches.
    omega_filename = os.path.join(outdir, "omega.npy")
    if os.path.exists(omega_filename):
      existing_omega_arr = np.load(omega_filename)
      if not np.allclose(omega_arr, existing_omega_arr):
        raise ValueError("omega file exists and does not match --omega")
    else:
      np.save(omega_filename, omega_arr)

    # Generate the coefficients array for each ell.
    for ell in ell_arr:
      logging.info(f"ell = {ell}")
      filename = os.path.join(outdir, f"ell{ell}.npy")
      if os.path.exists(filename) and not FLAGS.overwrite:
        logging.info(f"File already exists, skipping: {filename}")
        continue

      table = np.zeros((len(omega_arr), n_max), dtype=np.complex128)
      for i, omega in enumerate(omega_arr):
        logging.log_every_n_seconds(
            logging.INFO,
            f"omega = {omega:.4g} ({i+1} out of {len(omega_arr)})", 30)
        table[i] = coeff_loader(modename, ell, omega)

      np.save(filename, table)
      logging.info(f"Wrote {filename}")


if __name__ == "__main__":
  app.run(main)
