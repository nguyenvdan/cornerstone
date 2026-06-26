# Architecture

Cornerstone is built in phases. Each layer is usable on its own and feeds the
next. This document covers what exists today (Phases 0–1) and where later phases
plug in.

## Repository layout

```
cornerstone/
├── data/
│   ├── raw/          # cached scraped HTML (git-ignored, regenerable)
│   ├── interim/      # scratch / build logs (git-ignored)
│   └── processed/    # COMMITTED reproducible dataset (.parquet + .csv)
├── pipelines/        # Phase 1: fetch → parse → clean → join → version
│   ├── config.py         # universe, paths, rate limits, tier thresholds
│   ├── fetch.py          # polite cached HTTP (rate-limit + retry + cache)
│   ├── parse.py          # HTML helpers (lifts comment-wrapped tables)
│   ├── sources/bbref.py  # draft / NBA player / college scrapers
│   ├── features.py       # pre-draft feature row (no leakage)
│   ├── outcomes.py       # realized outcomes + tier label
│   ├── build_dataset.py  # orchestrator + CLI (the Phase 1 entry point)
│   ├── dybantsa.py       # AJ Dybantsa's profile row
│   └── data_dictionary.md
├── models/  agent/  eval/  api/  frontend/  notebooks/   # Phases 2–7 (stubs)
└── tests/            # pipeline unit tests (no network)
```

## Phase 1 data flow

```
DRAFT_YEARS (2003-2022)
        │
        ▼
  parse_draft(year)              # /draft/NBA_{year}.html
   → DraftPick[]                 #   pick, team, player_id, college, CAREER ws/bpm/vorp
        │
        ▼  for each pick
  parse_player(player_id)        # /players/{x}/{id}.html
   → PlayerPage                  #   birthdate, height, weight, cbb_id,
        │                        #   per-season advanced (WS/BPM/VORP/PER/USG)
        ▼  if cbb_id
  parse_college_final_season()   # /cbb/players/{id}.html
   → dict                        #   final college season (per-game + advanced)
        │
        ├──► build_features()    # pre-draft, no leakage  → prospect_features
        └──► build_outcome()     # realized + tier label  → realized_outcomes
                                          │
                                          ▼
                         join on player_id → prospects.{parquet,csv}
```

### Key design choices

- **Cache-first fetching.** Every URL is cached to `data/raw/` on first fetch.
  Re-runs are offline + deterministic, and partial runs resume for free. The
  cleaned dataset is committed; the raw cache is not.
- **Join on stable ids, not names.** The BBRef NBA `player_id` (parsed from the
  draft-page link) is the join key; the college page is reached via the
  BBRef→SRef link on the player page. This sidesteps accent/suffix/name-collision
  bugs (e.g. *Dončić*, *Jr./III*).
- **Comment-table handling.** BBRef hides most stat tables inside HTML comments
  to deter scrapers; `parse.make_soup` re-inflates them so every table is
  queryable.
- **No leakage by construction.** `features.py` only emits draft-time-available
  fields (draft slot, age, measurements, college production). Outcomes live in a
  separate table. This separation is what makes the Phase 4 back-test honest.
- **Final college season as the profile.** The most draft-relevant sample;
  avoids blending freshman and senior production.

## How later phases attach

| Phase | Reads | Produces |
|-------|-------|----------|
| 2 Comparables | `prospect_features` | `get_comparables(prospect)` → ranked analogs + feature attributions |
| 3 Projection | comparables + `realized_outcomes` | `project(prospect)` → tier distribution + season curve w/ intervals |
| 4 Eval | everything, draft-time only | calibration plot + headline metric vs. draft-slot baseline |
| 5 Roster fit | projection + roster data | `evaluate_fit(roster, projection)` → gaps + complementary archetypes |
| 6 Agent | all of the above as **tools** | autonomous multi-step scouting/strategy report |
| 7 Interface | precomputed + API | React app on a public URL |

The Phase 2–6 functions are designed to be the **tools** the Phase 6 agent
calls, so the agent layer is orchestration over a stable, tested toolbelt.
