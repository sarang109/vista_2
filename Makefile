.PHONY: install test smoke full data
install:
	python3 -m venv .venv && .venv/bin/pip install -q -e ".[test,data]"
test:
	.venv/bin/python -m pytest
data:
	./scripts/fetch_data.sh
smoke:
	.venv/bin/python -m vista2.run_all --config configs/smoke.json
full:
	.venv/bin/python -m vista2.run_all --config configs/full.json
