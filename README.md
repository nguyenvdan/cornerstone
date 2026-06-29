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
| 3 | Probabilistic outcome model | ✅ done |
| 4 | Back-testing / calibration | ✅ done |
| 5 | Roster-fit engine | ✅ done |
| 6 | Agentic orchestration | ✅ done |
| 7 | React frontend + API | ✅ done |
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
make project               # Dybantsa's probabilistic projection (Phase 3)
make backtest              # leakage-free back-test + calibration plot (Phase 4)
make skills                # scrape current-season NBA skill profiles (Phase 5)
make roster-fit            # Wizards roster fit around Dybantsa (Phase 5)
make agent                 # run the agent on the default Wizards question (Phase 6)
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

## Back-test results (Phase 4)

Leakage-aware **expanding-window** validation: every draft class is projected
using **only players drafted before it**, then compared to what actually
happened. The same protocol scores a **draft-position baseline** (how players
taken near each slot really turned out). The scouting model's hyperparameters
were **tuned on 2010–2015 prospects and validated on held-out 2016–2019
classes** (`make tune`) — proven to predict, not eyeballed to flatter AJ.
Reproduce with `make backtest`.

**459 prospects across 10 draft classes (2010–2019):**

| Metric | Scouting model | Draft-position baseline | Combined |
|--------|:---:|:---:|:---:|
| Within-1-tier accuracy | **0.75** | 0.71 | — |
| P(all-star+) AUC | **0.78** | 0.76 | 0.75 |
| P(all-star+) calibration (ECE ↓) | **0.023** | 0.029 | 0.026 |
| Career-VORP ranking (Spearman ↑) | **0.38** | 0.36 | 0.37 |

The scouting model is **numerically ahead** of the draft-position baseline on
every metric — but, scored honestly with a paired bootstrap, the gap is **not
statistically significant**: ΔAUC +0.018 (95% CI −0.009 to +0.045), ΔSpearman
+0.020 (CI −0.015 to +0.057). Both intervals straddle zero. We report this
rather than rounding it into a "win" (`make backtest` prints the CIs).

**Where the model *does* demonstrably add value — within draft-pick bands.** The
draft-position baseline is nearly flat inside a narrow band, so any
discrimination there is signal the *profile* contributes. Splitting star-AUC by
pick:

| Picks | n | Model AUC | Baseline AUC | Δ |
|------|:--:|:--:|:--:|:--:|
| 1–5 | 43 | **0.66** | 0.55 | +0.11 |
| 15–30 | 131 | **0.59** | 0.44 | +0.15 |
| 31–60 | 202 | **0.58** | 0.52 | +0.06 |

This is the real result: the model separates players picked *near each other*,
which is exactly the job — the overall AUC just washes it out because base rates
differ across bands.

The generalizable levers (draft capital + strength of competition) carry the
signal; **recruiting rank, conference, age-vs-class and combine athleticism were
all tested and did *not* improve out-of-sample** prediction — draft capital
already encodes them. Two further **honest negative results** from the bake-off:

- **Combining** profile + draft position (naive average, a leakage-free
  walk-forward logistic *stack*, and isotonic recalibration — `eval/stacker.py`)
  does **not** beat the standalone profile model out-of-sample. We keep the
  principled stacker but report the standalone model.
- A **directly-supervised GBM** (`make supervised`) on the same features
  *underperforms* the kNN (AUC 0.74 vs 0.78, significantly) — with ~460 matured
  examples and a 15% star rate, the kNN's local averaging wins. It also barely
  changes the **range compression** below, confirming that's not a kNN artifact.

![Calibration and accuracy vs baseline](eval/calibration.png)

**Known weakness — the point estimate compresses the range.** `expected_career_vorp`
is a similarity-weighted *mean*, so it cannot reach the tails: eventual
superstars get a predicted ~11 career VORP (they average ~40). This is inherent
to any conditional-*mean* predictor (the GBM does it too). The honest signal for
upside lives in the **distribution's upper quantiles**, which the model already
computes and which *do* separate the tiers (superstar p90 ≈ 31 vs bust p90 ≈ 11)
— so feature the `career_vorp_band` p90/p95, not the mean, for "ceiling".

**Limitations** (stated honestly): small early-cohort samples (only ~15
superstars in the test window, so high-confidence calibration is barely
measurable); survivorship and era effects; career VORP is cumulative, so it
rewards longevity (corr 0.65 with games played) and is era-confounded; training
uses earlier cohorts' eventually-realized careers (fair to both model and
baseline). See `eval/backtest.py`, `eval/supervised.py`, `eval/tune_context.py`.

## Web app (Phase 7)

A clean, mobile-friendly React (Vite + TypeScript) dashboard with five views —
**Projection**, **Comparables**, **Wizards build**, **Ask the agent**, and
**Methodology**. It's fully static: it reads the precomputed JSON artifacts, so
it deploys as a static site (Vercel-ready) with no backend.

```bash
make frontend-dev     # http://localhost:5173
make frontend-build   # production build -> frontend/dist/
```

Deploy: import the repo in Vercel, set **Root Directory** to `frontend` (Vite
auto-detected). See [`frontend/README.md`](frontend/README.md). _Live URL: add
after first deploy._

## The agent (Phase 6)

The whole system is exposed as a toolbelt (`lookup_prospect`, `find_comparables`,
`project_development`, `evaluate_roster_fit`, `team_skill_summary`) wrapped in an
agent that answers a natural-language question with a multi-step, **cited**
scouting & strategy report. Run with `make agent` (or pass your own question:
`uv run python -m agent.runner "How will AJ Dybantsa develop?"`).

Two modes over the same tools:
- **Autonomous** — with `ANTHROPIC_API_KEY` set (and `uv sync --extra agent`), an
  LLM plans and chooses tool calls itself, then synthesizes the report. Easiest
  setup: `cp .env.example .env` and paste your key into the git-ignored `.env` —
  the runner auto-loads it (via `python-dotenv`). An invalid/absent key falls
  back to scripted mode instead of erroring.
- **Scripted** — with no key, a deterministic orchestrator routes the request to
  the relevant tools and synthesizes the report from their outputs, so the agent
  is fully demonstrable offline.

Both emit an inspectable reasoning trace, and every number in the report is read
from a tool output — the agent never fabricates stats and always states
uncertainty as a distribution.

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
