# cornerstone

**An agentic, uncertainty-aware player-development and roster-fit projection system, built around AJ Dybantsa and the Washington Wizards rebuild.**

Cornerstone is a *decision-support-under-uncertainty* system. The product
question it answers: **"AJ Dybantsa is the Wizards' new cornerstone — how is he
likely to develop, and how should the team build around him?"**

It foregrounds four skills that are distinct from typical RAG/retrieval work:

1. **Agentic orchestration** — an autonomous agent that plans a multi-step
   analysis and calls the models below as tools _(Phase 6)_.
2. **Probabilistic modeling with uncertainty** — outcomes as calibrated
   distributions, not point predictions _(Phase 3)_.
3. **Rigorous back-testing** — leakage-aware validation against real draft
   history with calibration metrics _(Phase 4)_.
4. **Quantitative modeling** — embedding-based historical comparables and
   trajectory modeling _(Phase 2)_.

---

## Architecture (target)

```
            ┌─────────────────────────────────────────────────────┐
            │                   AGENT (Phase 6)                     │
            │  plans → calls tools → interprets → synthesizes report│
            └───────────┬───────────────┬───────────────┬──────────┘
                        │               │               │
              get_comparables       project        evaluate_fit
              (Phase 2)           (Phase 3)         (Phase 5)
                        │               │               │
            ┌───────────┴───────────────┴───────────────┴──────────┐
            │        Phase 1 data pipeline → versioned dataset      │
            │  BBRef / SRef  →  fetch+cache → parse → clean → join  │
            └───────────────────────────────────────────────────────┘
                         back-tested by  Eval (Phase 4)
                         served through   API + React (Phase 7)
```

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for module responsibilities and data flow.

## Build status

| Phase | What | Status |
|-------|------|--------|
| 0 | Scaffolding | ✅ done |
| 1 | Data pipeline (fetch → clean → join → version) | ✅ done |
| 2 | Comparables engine (embedding similarity) | ✅ done |
| 3 | Probabilistic outcome model | ⏳ planned |
| 4 | Back-testing / calibration | ⏳ planned |
| 5 | Roster-fit engine | ⏳ planned |
| 6 | Agentic orchestration | ⏳ planned |
| 7 | React frontend + API | ⏳ planned |
| 8 | Polish & writeup | ⏳ planned |

---

## Quickstart

Requires Python ≥ 3.11 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev        # install
make data-sample           # quick 2-year build (~2 min) to verify the pipeline
make data                  # full 2003-2022 universe (first run ~1.5-2 hr; cached after)
make dybantsa              # build AJ Dybantsa's pre-draft profile row
make comparables           # Dybantsa's top historical analogs (Phase 2)
make test                  # unit tests
```

Outputs land in `data/processed/` as committed `.parquet` + `.csv`:

| File | Description |
|------|-------------|
| `prospect_features.*` | Pre-draft features, **no leakage** (draft slot, age, measurements, final college season). |
| `realized_outcomes.*` | Realized NBA outcomes (WS/BPM/VORP, early-career trajectory) + a tier label. |
| `prospects.*` | The two joined on `player_id`. |
| `dybantsa.*` | AJ Dybantsa's row, identical schema to `prospect_features`. |

Full column reference: [`pipelines/data_dictionary.md`](pipelines/data_dictionary.md).

### Reproducibility

`data/raw/` (cached HTML) is git-ignored and regenerable; the cleaned
`data/processed/` dataset is committed, so the project is reproducible offline.
Re-running any build hits the cache and is deterministic.

## Data sources

Public, scraped politely (≤ ~18 req/min, cached aggressively):

- [Basketball Reference](https://www.basketball-reference.com) — drafts, NBA
  careers, advanced stats.
- [Sports Reference (College)](https://www.sports-reference.com/cbb/) —
  pre-draft college production.

## Honesty guardrails

No false precision (projections are explicitly probabilistic), no leakage in the
back-test (draft-time-available data only), and openly stated limitations
(sample size, survivorship, era effects). See the data dictionary's limitations
section.

## License

MIT — see [LICENSE](LICENSE).
