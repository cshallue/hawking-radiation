import os

from absl import logging

from hawkrad.kg_modes.asymptotic.gen_coeffs import (CoeffGenerator,
                                                    generate_b_coeffs)
from hawkrad.kg_modes.asymptotic.load_coeffs import read_coeffs
from hawkrad.kg_modes.asymptotic.rho_fns import RhoFnInHoriz, RhoFnUpInf
from hawkrad.utils import Timer


def load_asymptotic_solution(coeff_loader, modename, ell, omega):
  coeffs = coeff_loader(modename, ell, omega)
  if modename == "in":
    return RhoFnInHoriz(omega, coeffs)
  if modename == "up":
    return RhoFnUpInf(omega, coeffs)


class CoeffLoader:

  def __init__(self,
               coeffs_table_dir=None,
               up_def_filename=None,
               n_coeffs=None):
    self.coeffs_table_dir = coeffs_table_dir
    self.n_coeffs = n_coeffs

    self.up_coeff_gen = None
    if up_def_filename is not None:
      with Timer() as t:
        self.up_coeff_gen = CoeffGenerator(up_def_filename, n_coeffs)
      logging.info(f"Loaded up mode coefficient file in {t.elapsed:.2g}s")

  def __call__(self, modename, ell, omega):
    # First try to read the coefficients from a coefficient file.
    coeff_dir = (os.path.join(self.coeffs_table_dir, modename)
                 if self.coeffs_table_dir else "")
    try:
      return read_coeffs(coeff_dir, ell, omega, self.n_coeffs)
    except (FileNotFoundError, KeyError):
      pass

    # Okay, we need to generate the coefficients.
    if modename == "up":
      if self.up_coeff_gen is None:
        raise KeyError(
            f"(ell, omega) = ({ell:d}, {omega:.4g}) not found in table and "
            "up coefficient generator is not defined")
      return self.up_coeff_gen(omega, ell)

    if modename != "in":
      raise ValueError(modename)

    if self.n_coeffs is None:
      raise ValueError("n_coeffs is required to generate in modes")

    return generate_b_coeffs(omega, ell, self.n_coeffs)
