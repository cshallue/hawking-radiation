import os

import numpy as np


def _index_of(arr, val):
  idxs = np.where(np.isclose(arr, val, rtol=1e-10, atol=1e-10))[0]
  if len(idxs) != 1:
    raise KeyError(val)
  return idxs[0]


def _read_coeffs_from_npy(coeff_dir, ell, omega):
  # First find the row index of the coeff table.
  omega_arr = np.load(os.path.join(coeff_dir, "omega.npy"))
  i = _index_of(omega_arr, omega)
  # Now load the coeff table and return that row.
  table = np.load(os.path.join(coeff_dir, f"ell{ell}.npy"))
  return table[i]


def _read_coeffs_from_csv(coeff_dir, ell, omega):
  filename = os.path.join(coeff_dir, f"ell{ell}.csv")
  with open(filename, "r") as f:
    for i, line in enumerate(f):
      key, values = line.split(",", 1)
      if i == 0:
        assert key == "omega*2M"
        continue
      if np.isclose(float(key), omega, rtol=1e-10, atol=1e-10):
        return np.complex128(values.split(","))
  raise KeyError(omega)


def read_coeffs(coeff_dir, ell, omega, n_max=None):
  try:
    coeffs = _read_coeffs_from_npy(coeff_dir, ell, omega)
  except FileNotFoundError:
    coeffs = _read_coeffs_from_csv(coeff_dir, ell, omega)

  bad_i = np.where(~np.isfinite(coeffs))[0]  # Catch overflow or nans.
  if len(bad_i > 0):
    # Take only coeffs up to the first bad index.
    coeffs = coeffs[:bad_i[0]]
  if n_max:
    coeffs = coeffs[:n_max + 1]
  return coeffs


def _load_omegas_from_csv(filename):
  omegas = []
  with open(filename, "r") as f:
    for i, line in enumerate(f):
      key = line.split(",", 1)[0]
      if i == 0:
        assert key == "omega*2M"
        continue
      omegas.append(float(key))

  return np.array(omegas)


def load_omegas(coeff_dir,
                ell=0,
                modename="up",
                omega_min=None,
                omega_max=None):
  try:
    omega = np.load(os.path.join(coeff_dir, modename, f"omega.npy"))
  except FileNotFoundError:
    omega = _load_omegas_from_csv(
        os.path.join(coeff_dir, modename, f"ell{ell}.csv"))

  omega_min = omega_min or -np.inf
  omega_max = omega_max or np.inf
  mask = (omega >= omega_min) & (omega <= omega_max)
  return omega[mask]
