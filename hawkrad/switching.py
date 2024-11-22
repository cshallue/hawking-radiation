import numpy as np


# As defined in Ng (2022).
class ChiFn:

  def __init__(self, tau_mid, delta):
    self.tau_mid = tau_mid
    self.delta = delta

  def __call__(self, tau):
    delta_tau = tau - self.tau_mid
    return np.where(
        np.abs(delta_tau) < self.delta,
        np.cos(np.pi / (2 * self.delta) * delta_tau)**4, 0)

  @property
  def tau0(self):
    return self.tau_mid - self.delta

  @property
  def tau1(self):
    return self.tau_mid + self.delta
