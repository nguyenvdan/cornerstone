"""Draft-position baseline.

The naive predictor everyone has for free: a prospect's draft slot. For a given
pick we look at how *training* players taken near that slot actually turned out.
This is the bar the profile model must clear to demonstrate it adds value.

Implemented as a smooth kernel over pick distance (bandwidth in picks), so the
estimate degrades gracefully and isn't jumpy at bucket edges.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipelines import config


class DraftPositionBaseline:
    def __init__(self, bandwidth: float = 5.0) -> None:
        self.bandwidth = bandwidth

    def fit(self, train: pd.DataFrame) -> DraftPositionBaseline:
        t = train[train["career_vorp"].notna() & train["draft_pick"].notna()]
        self.picks = t["draft_pick"].to_numpy(float)
        self.vorp = t["career_vorp"].to_numpy(float)
        self.star = t["outcome_tier"].isin(["all_star", "superstar"]).to_numpy(int)
        self.tiers = t["outcome_tier"].to_numpy()
        return self

    def _weights(self, pick: float) -> np.ndarray:
        return np.exp(-0.5 * ((self.picks - pick) / self.bandwidth) ** 2)

    def expected_vorp(self, pick: float) -> float:
        w = self._weights(pick)
        return float(np.average(self.vorp, weights=w))

    def p_star_plus(self, pick: float) -> float:
        w = self._weights(pick)
        return float(np.average(self.star, weights=w))

    def tier_probabilities(self, pick: float) -> dict[str, float]:
        w = self._weights(pick)
        probs = {tier: 0.0 for tier in config.TIER_ORDER}
        for tier, weight in zip(self.tiers, w, strict=False):
            probs[tier] += weight
        total = sum(probs.values())
        return {t: p / total for t, p in probs.items()}
