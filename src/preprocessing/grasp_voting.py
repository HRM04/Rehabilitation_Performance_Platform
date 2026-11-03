"""Temporal label averaging for grasp classification.

Replaces a hand's current grasp label with the most common label over its last
few frames when one label has at least `min_votes` of them. Short flicker is
removed, genuine changes still come through, and AMBIGUOUS or UNKNOWN are kept
as honest labels rather than hidden.

Window length: GRASP_VOTE_FRAMES (default 5). Set to 1 to switch it off.
"""
import os
from collections import deque


class GraspVoter:
    def __init__(self, window=None, max_age_ms=1500, min_votes=3):
        if window is None:
            try:
                window = int(os.environ.get("GRASP_VOTE_FRAMES", "5"))
            except ValueError:
                window = 5
        self.window = window
        self.max_age_ms = max_age_ms
        self.min_votes = min_votes
        self._hist = {}

    def describe(self):
        return {"window": self.window, "min_votes": self.min_votes, "max_age_ms": self.max_age_ms,
                "enabled": self.window > 1}

    def reset(self, hand=None):
        if hand is None:
            self._hist.clear()
        else:
            self._hist.pop(hand, None)

    def vote(self, hand, t_ms, label, labels):
        """Return (label, candidate_list) after voting. Input lists are not modified."""
        labels = list(labels or [])
        if self.window <= 1:
            return label, labels
        h = self._hist.setdefault(hand, deque())
        h.append((t_ms, label, labels))
        while h and (t_ms - h[0][0] > self.max_age_ms or len(h) > self.window):
            h.popleft()
        counts = {}
        for _, lab, _ in h:
            counts[lab] = counts.get(lab, 0) + 1
        best = max(counts, key=lambda k: (counts[k], k == label))
        if counts[best] >= self.min_votes and best != label:
            for _, lab, cands in reversed(h):
                if lab == best:
                    return lab, list(cands)
        return label, labels


GRASP_VOTER = GraspVoter()
