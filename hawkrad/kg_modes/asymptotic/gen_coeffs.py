import numpy as np
import sympy
from absl import logging
from sympy import abc
from sympy.parsing.sympy_parser import parse_expr


def _as_rational(omega):
  denom_lim = int(1e6) if omega <= 0.1 else 100
  return sympy.Rational(omega).limit_denominator(denom_lim)


class CoeffGenerator:

  def __init__(self, filename, n_max=None):
    if n_max is None:
      n_max = np.inf
    logging.info(f"Reading coefficient functions from {filename}")
    expr_strs = []
    with open(filename, "r") as f:
      for n, line in enumerate(f.readlines()):
        expr_strs.append(line.strip('"\n').replace("^", "**"))
        if n >= n_max:
          break

    self.exprs = [parse_expr(e) for e in expr_strs]
    self.n_max = len(self.exprs)
    logging.info(f"Parsed {self.n_max} coefficient functions")

  def __call__(self, omega, ell):
    subs = {
        abc.omega: _as_rational(omega),
        abc.L: sympy.Integer(ell * (ell + 1))
    }
    # sympy.lambdify would make this faster, but it evaluates numerically rather
    # than symbolically and dramatically reduces the precision.
    return np.array([expr.evalf(subs=subs) for expr in self.exprs],
                    dtype=np.complex128)


def generate_b_coeffs(omega, ell, n_max):
  omega = _as_rational(omega)
  L = sympy.Integer(ell * (ell + 1))
  I = sympy.I

  b = [sympy.S.One]
  for n in range(1, n_max + 1):
    x1 = 6 * I * (n - 1) * omega - 2 * n**2 + 5 * n + L - 2
    x2 = 6 * I * (n - 2) * omega - n**2 + 5 * n + L - 6
    x3 = 2 * I * (n - 3) * omega
    denom = n * (n - 2 * I * omega)
    if n == 1:
      bn = (x1 * b[n - 1]) / denom
    elif n == 2:
      bn = (x1 * b[n - 1] + x2 * b[n - 2]) / denom
    else:
      bn = (x1 * b[n - 1] + x2 * b[n - 2] + x3 * b[n - 3]) / denom

    # Keep all terms symbolic, simplifying to a complex fraction each time.
    b.append(bn.simplify())

  return np.array([bn.evalf() for bn in b], dtype=np.complex128)
