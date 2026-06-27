"""Embedding-based historical comparables engine (Phase 2).

Given a prospect's pre-draft profile, find the most similar historical
prospects and explain *why* they match.

Method
------
1. Restrict to the college universe (the profile is college production), keep
   rows with the core stats present.
2. Standardize the 16-feature vector (z-scores) -> the prospect "embedding".
3. Nearest-neighbor search in that standardized space (Euclidean).
4. Explain each match by decomposing the squared distance per feature: the
   features with the smallest z-gap are *why they're alike*; the largest gaps
   are *where they differ*.

A 2-D PCA projection of the embedding is also exposed for later visualization.

Used as the ``get_comparables`` tool by the Phase 6 agent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from pipelines import config

from .features import (
    FEATURE_DISPLAY,
    META_COLUMNS,
    SIMILARITY_FEATURES,
    format_value,
)


@dataclass
class FeatureDelta:
    feature: str
    display: str
    prospect_value: str
    comp_value: str
    z_gap: float          # |z_prospect - z_comp| (0 = identical, in std devs)
    direction: str        # comp is "higher" / "lower" / "similar"


@dataclass
class Comparable:
    player_id: str
    player_name: str
    draft_year: int
    draft_pick: int | None
    position: str | None
    outcome_tier: str
    career_vorp: float | None
    similarity: float                 # 0-100 match score
    distance: float                   # raw Euclidean distance in z-space
    most_alike: list[FeatureDelta]
    biggest_gaps: list[FeatureDelta]

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["most_alike"] = [fd.__dict__ for fd in self.most_alike]
        d["biggest_gaps"] = [fd.__dict__ for fd in self.biggest_gaps]
        return d


class ComparablesEngine:
    """Fit on the historical universe, then query with any prospect row."""

    def __init__(self) -> None:
        self.scaler_: StandardScaler | None = None

    def fit(self, prospects: pd.DataFrame) -> ComparablesEngine:
        college = prospects[prospects["has_college_stats"]].copy()
        college = college[college["coll_pts_per_g"].notna()]
        college = college.drop_duplicates(subset="player_id").set_index("player_id")

        raw = college[SIMILARITY_FEATURES].astype(float)
        self.medians_ = raw.median()
        raw = raw.fillna(self.medians_)

        self.raw_ = raw                                    # imputed, unscaled
        self.meta_ = college[META_COLUMNS]
        self.scaler_ = StandardScaler().fit(raw.values)
        self.Z_ = self.scaler_.transform(raw.values)      # (n, f)
        self.ids_ = raw.index.to_numpy()

        # Scale at which the match score halves: the median pairwise distance.
        dmat = pairwise_distances(self.Z_)
        iu = np.triu_indices_from(dmat, k=1)
        self.d_half_ = float(np.median(dmat[iu]))

        # 2-D embedding for later visualization.
        self.pca_ = PCA(n_components=2).fit(self.Z_)
        self.embedding_ = self.pca_.transform(self.Z_)
        return self

    # -- internals ---------------------------------------------------------
    def _vectorize(self, prospect: pd.Series) -> np.ndarray:
        raw = prospect.reindex(SIMILARITY_FEATURES).astype(float)
        raw = raw.fillna(self.medians_)
        return self.scaler_.transform(raw.values.reshape(1, -1))[0]

    def _score(self, distance: float) -> float:
        # 100 at distance 0, 50 at the median pairwise distance, smooth decay.
        return round(100.0 * 2.0 ** (-distance / self.d_half_), 1)

    def _deltas(self, prospect_raw: pd.Series, comp_id: str,
                z_prospect: np.ndarray, z_comp: np.ndarray) -> list[FeatureDelta]:
        comp_raw = self.raw_.loc[comp_id]
        out: list[FeatureDelta] = []
        for j, feat in enumerate(SIMILARITY_FEATURES):
            gap = abs(z_prospect[j] - z_comp[j])
            if z_comp[j] - z_prospect[j] > 0.25:
                direction = "higher"
            elif z_comp[j] - z_prospect[j] < -0.25:
                direction = "lower"
            else:
                direction = "similar"
            out.append(
                FeatureDelta(
                    feature=feat,
                    display=FEATURE_DISPLAY[feat],
                    prospect_value=format_value(feat, prospect_raw.get(feat)),
                    comp_value=format_value(feat, comp_raw[feat]),
                    z_gap=round(float(gap), 2),
                    direction=direction,
                )
            )
        return out

    # -- public API --------------------------------------------------------
    def get_comparables(
        self,
        prospect: pd.Series,
        k: int = 10,
        exclude_player_id: str | None = None,
        n_explain: int = 3,
    ) -> list[Comparable]:
        if self.scaler_ is None:
            raise RuntimeError("ComparablesEngine.fit() must be called first")

        z = self._vectorize(prospect)
        prospect_raw = prospect.reindex(SIMILARITY_FEATURES).astype(float).fillna(self.medians_)
        distances = np.linalg.norm(self.Z_ - z, axis=1)
        order = np.argsort(distances)

        results: list[Comparable] = []
        for idx in order:
            pid = self.ids_[idx]
            if exclude_player_id is not None and pid == exclude_player_id:
                continue
            deltas = self._deltas(prospect_raw, pid, z, self.Z_[idx])
            ranked = sorted(deltas, key=lambda d: d.z_gap)
            meta = self.meta_.loc[pid]
            pick = None if pd.isna(meta["draft_pick"]) else int(meta["draft_pick"])
            vorp = None if pd.isna(meta["career_vorp"]) else float(meta["career_vorp"])
            results.append(
                Comparable(
                    player_id=str(pid),
                    player_name=meta["player_name"],
                    draft_year=int(meta["draft_year"]),
                    draft_pick=pick,
                    position=meta["position"],
                    outcome_tier=meta["outcome_tier"],
                    career_vorp=vorp,
                    similarity=self._score(float(distances[idx])),
                    distance=round(float(distances[idx]), 3),
                    most_alike=ranked[:n_explain],
                    biggest_gaps=list(reversed(ranked))[:2],
                )
            )
            if len(results) >= k:
                break
        return results


# --------------------------------------------------------------------------
# CLI: build Dybantsa's comparables + a sanity check, write JSON for the app.
# --------------------------------------------------------------------------
def _print_comp(rank: int, c: Comparable) -> None:
    pick = f"#{c.draft_pick}" if c.draft_pick else "—"
    print(f"  {rank:2d}. {c.player_name:24s} ({c.draft_year}, {pick})  "
          f"sim {c.similarity:5.1f}  outcome={c.outcome_tier}  VORP={c.career_vorp}")
    alike = ", ".join(f"{d.display} ({d.comp_value})" for d in c.most_alike)
    gaps = ", ".join(f"{d.display} {d.direction} ({d.comp_value} vs {d.prospect_value})"
                     for d in c.biggest_gaps)
    print(f"      alike: {alike}")
    print(f"      differs: {gaps}")


def main() -> int:
    prospects = pd.read_parquet(config.PROCESSED / "prospects.parquet")
    engine = ComparablesEngine().fit(prospects)
    print(f"Fitted comparables engine on {len(engine.ids_)} college prospects "
          f"({len(SIMILARITY_FEATURES)} features).\n")

    dyb = pd.read_parquet(config.PROCESSED / "dybantsa.parquet").iloc[0]
    comps = engine.get_comparables(dyb, k=10)
    print("AJ Dybantsa — top 10 historical comparables:")
    for i, c in enumerate(comps, 1):
        _print_comp(i, c)

    out = {
        "prospect": "AJ Dybantsa",
        "n_universe": len(engine.ids_),
        "features": SIMILARITY_FEATURES,
        "comparables": [c.to_dict() for c in comps],
    }
    path = config.PROCESSED / "dybantsa_comparables.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {path}")

    # Sanity check: a known player's comps should be the right archetype.
    davis = prospects[prospects["player_name"] == "Anthony Davis"]
    if not davis.empty:
        print("\nSanity check — Anthony Davis comparables (excluding himself):")
        for i, c in enumerate(engine.get_comparables(davis.iloc[0], k=5,
                              exclude_player_id=davis.iloc[0]["player_id"]), 1):
            _print_comp(i, c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
