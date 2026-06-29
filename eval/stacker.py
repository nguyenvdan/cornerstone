"""Walk-forward stacked + calibrated ensemble of the two predictors.

The old "combined" model was a fixed ``0.5 * profile + 0.5 * draft_position``
average. The back-test showed that average was the *only* combination that beat
the baseline significantly — but it **degraded calibration** (ECE up vs the
profile model alone), because blindly averaging two probabilities is not a
calibrated operation.

This replaces it with a proper **stack**: a logistic meta-model learns *how
much* to trust each predictor for the star call, and a linear blend does the
same for expected VORP. An optional isotonic layer recalibrates the stacked
probability. Everything is fit **walk-forward** — for draft class Y the stacker
sees only predictions on classes *before* Y — so there is no leakage, matching
the expanding-window protocol of the back-test itself. Until enough history has
accrued, it falls back to the simple average, so early classes still get a
sane combined number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, LogisticRegression

# Columns the stacker reads from a back-test prediction frame.
_STAR_X = ["model_p_star", "base_p_star"]
_VORP_X = ["model_exp_vorp", "base_exp_vorp"]


class WalkForwardStacker:
    """Leakage-free stack of the profile model and the draft-position baseline.

    ``min_train`` past prospects are required before the learned stack kicks in;
    below that we average. ``calibrate`` adds an isotonic recalibration of the
    stacked star probability once ``min_calibrate`` positives are available.
    """

    def __init__(self, min_train: int = 60, min_calibrate: int = 150,
                 min_positives: int = 15, calibrate: bool = True) -> None:
        self.min_train = min_train
        self.min_calibrate = min_calibrate
        self.min_positives = min_positives
        self.calibrate = calibrate
        self.star_clf: LogisticRegression | None = None
        self.vorp_reg: LinearRegression | None = None
        self.iso: IsotonicRegression | None = None
        self.fitted = False

    def fit(self, past: pd.DataFrame) -> WalkForwardStacker:
        """Fit on past test-year predictions (``model_*``/``base_*`` + actuals)."""
        self.star_clf = self.vorp_reg = self.iso = None
        self.fitted = False
        if past is None or len(past) < self.min_train:
            return self
        y = past["actual_star"].to_numpy(int)
        if y.sum() < 2 or y.sum() > len(y) - 2:   # need both classes present
            return self

        Xs = past[_STAR_X].to_numpy(float)
        self.star_clf = LogisticRegression(max_iter=1000).fit(Xs, y)
        Xv = past[_VORP_X].to_numpy(float)
        self.vorp_reg = LinearRegression().fit(Xv, past["actual_vorp"].to_numpy(float))

        if self.calibrate and len(past) >= self.min_calibrate and y.sum() >= self.min_positives:
            p = self.star_clf.predict_proba(Xs)[:, 1]
            self.iso = IsotonicRegression(out_of_bounds="clip").fit(p, y)
        self.fitted = True
        return self

    def predict(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (combined P(star+), combined expected VORP) for ``rows``.

        Falls back to the plain average whenever the stack isn't fitted yet."""
        mp, bp = rows["model_p_star"].to_numpy(float), rows["base_p_star"].to_numpy(float)
        mv, bv = rows["model_exp_vorp"].to_numpy(float), rows["base_exp_vorp"].to_numpy(float)
        avg_p, avg_v = 0.5 * (mp + bp), 0.5 * (mv + bv)
        if not self.fitted:
            return avg_p, avg_v
        p = self.star_clf.predict_proba(rows[_STAR_X].to_numpy(float))[:, 1]
        if self.iso is not None:
            p = self.iso.predict(p)
        v = self.vorp_reg.predict(rows[_VORP_X].to_numpy(float))
        return np.clip(p, 0.0, 1.0), v


def walk_forward_combine(res: pd.DataFrame, year_col: str = "draft_year",
                         **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Produce leakage-free combined predictions for every row of ``res``.

    Processes draft classes in chronological order; each class is predicted by a
    stacker fit only on *earlier* classes' predictions. Returns (p_star, exp_vorp)
    aligned to ``res``'s original row order.
    """
    p_out = np.empty(len(res), float)
    v_out = np.empty(len(res), float)
    seen: list[pd.DataFrame] = []
    stacker = WalkForwardStacker(**kwargs)
    for year in sorted(res[year_col].unique()):
        mask = (res[year_col] == year).to_numpy()
        stacker.fit(pd.concat(seen) if seen else None)
        p_out[mask], v_out[mask] = stacker.predict(res[mask])
        seen.append(res[mask])
    return p_out, v_out
