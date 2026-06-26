.PHONY: install data data-sample test lint clean

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

test:
	uv run pytest -q

lint:
	uv run ruff check .

clean:
	rm -rf data/interim/* data/processed/*.parquet data/processed/*.csv
