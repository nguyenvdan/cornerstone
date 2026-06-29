.PHONY: install data data-sample backtest tune supervised test lint clean

install:
	uv sync --extra dev

# Full historical universe (2003-2022). First run scrapes BBRef politely
# (~30-45 min); subsequent runs hit the local cache and are fast.
data:
	uv run python -m pipelines.build_dataset

# Quick end-to-end smoke build on a 2-year subset (for development/CI).
data-sample:
	uv run python -m pipelines.build_dataset --years 2017 2018

dybantsa:
	uv run python -m pipelines.dybantsa

# Phase 2: fit the comparables engine and print Dybantsa's historical analogs.
comparables:
	uv run python -m models.comparables

# Phase 3: probabilistic development projection (tiers, curve, swing factors).
project:
	uv run python -m models.projection

# Phase 4: leakage-aware temporal back-test + calibration plot vs baseline.
backtest:
	uv run python -m eval.backtest

# Phase 5: scrape current-season NBA skill profiles (league-wide).
skills:
	uv run python -m pipelines.nba_skills

# Phase 5: roster-fit readout for the Wizards around Dybantsa.
roster-fit:
	uv run python -m models.roster_fit

# Combine athleticism: join public NBA combine data (wingspan/vertical) + AJ's line.
combine:
	uv run python -m pipelines.combine

# Peak per-36 production of the comparable universe (from cached player pages).
per36:
	uv run python -m pipelines.per36

# Enriched predictors: recruiting rank (RSCI), combine, conference SOS, age-vs-class.
recruiting:
	uv run python -m pipelines.recruiting
enrich:
	uv run python -m pipelines.enrich

# Tune the scouting model's hyperparameters on the back-test (not on AJ).
tune:
	uv run python -m eval.tune_context

# Phase 4+: directly-supervised GBM prototype vs the kNN model + baseline
# (significance CIs + range-compression check). An honest model bake-off.
supervised:
	uv run python -m eval.supervised

# Phase 6: run the agent. Scripted (no key) by default; autonomous LLM mode if
# ANTHROPIC_API_KEY is set and `uv sync --extra agent` has been run.
agent:
	uv run python -m agent.runner

# Phase 7: sync the precomputed JSON/plot artifacts into the frontend, so the
# deployed app is self-contained and static.
frontend-data:
	mkdir -p frontend/public/data
	cp data/processed/dybantsa_projection.json data/processed/dybantsa_comparables.json \
	   data/processed/wizards_fit.json data/processed/agent_report.json \
	   eval/backtest_results.json frontend/public/data/
	cp eval/calibration.png frontend/public/

# Phase 7: build the React app (run frontend-data first if models changed).
frontend-build: frontend-data
	cd frontend && npm install && npm run build

# Phase 7: run the frontend locally at http://localhost:5173
frontend-dev:
	cd frontend && npm install && npm run dev

test:
	uv run pytest -q

lint:
	uv run ruff check .

clean:
	rm -rf data/interim/* data/processed/*.parquet data/processed/*.csv
