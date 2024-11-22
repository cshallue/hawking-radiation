from timeit import default_timer


class Timer:

  def __init__(self):
    self._timer_on = False
    self._t_start_sec = 0
    self._elapsed_sec = 0

  def start(self):
    if self._timer_on:
      raise ValueError("Timer already started")

    self._t_start_sec = default_timer()
    self._timer_on = True

  def stop(self):
    if not self._timer_on:
      raise ValueError("Timer not started")

    t_end_sec = default_timer()
    self._elapsed_sec += (t_end_sec - self._t_start_sec)
    self._timer_on = False

  def clear(self):
    if self._timer_on:
      raise ValueError("Timer currently running")

    self._elapsed_sec = 0

  def __enter__(self):
    self.start()
    return self

  def __exit__(self, *args):
    self.stop()

  @property
  def elapsed(self):
    return self._elapsed_sec
