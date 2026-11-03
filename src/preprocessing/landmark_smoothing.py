"""Hand landmark smoothing (One Euro filter).

Reduces frame-to-frame jitter in MediaPipe hand landmarks while keeping
fast, deliberate movements responsive. Time-aware, so it copes with the
uneven processing rate (about 4 to 9 frames/s) of the capture loop.

Disable for a run with:  HAND_SMOOTHING=off python main.py
"""
import math
import os


class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=1.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.t_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        dt = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


class HandLandmarkSmoother:
    """Smooths x, y, z of the 21 landmarks separately for each hand."""

    def __init__(self, min_cutoff=1.0, beta=1.0, d_cutoff=1.0,
                 max_gap_s=1.0, enabled=None):
        if enabled is None:
            enabled = os.environ.get("HAND_SMOOTHING", "on").strip().lower() not in ("off", "0", "false", "no")
        self.enabled = enabled
        self.params = dict(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self.max_gap_s = max_gap_s
        self._filters = {}
        self._last_t = {}

    def describe(self):
        return {"method": "one_euro", "enabled": self.enabled,
                "max_gap_s": self.max_gap_s, **self.params}

    def reset(self, hand_label=None):
        if hand_label is None:
            self._filters.clear()
            self._last_t.clear()
            return
        for key in [k for k in self._filters if k[0] == hand_label]:
            del self._filters[key]
        self._last_t.pop(hand_label, None)

    def smooth_values(self, hand_label, index, axis, value, t_s):
        key = (hand_label, index, axis)
        flt = self._filters.get(key)
        if flt is None:
            flt = OneEuroFilter(**self.params)
            self._filters[key] = flt
        return flt(value, t_s)

    def smooth(self, hand_label, hand_landmarks, timestamp_ms):
        """Smooth a MediaPipe hand landmark list in place."""
        if not self.enabled:
            return hand_landmarks
        t_s = timestamp_ms / 1000.0
        last = self._last_t.get(hand_label)
        if last is not None and (t_s - last) > self.max_gap_s:
            self.reset(hand_label)
        self._last_t[hand_label] = t_s
        for i, lm in enumerate(hand_landmarks.landmark):
            lm.x = self.smooth_values(hand_label, i, "x", lm.x, t_s)
            lm.y = self.smooth_values(hand_label, i, "y", lm.y, t_s)
            lm.z = self.smooth_values(hand_label, i, "z", lm.z, t_s)
        return hand_landmarks


HAND_SMOOTHER = HandLandmarkSmoother()
