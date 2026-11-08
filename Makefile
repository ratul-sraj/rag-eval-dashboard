PY ?= python
OUT ?= artifacts/demo_run
RUN_ID ?= demo-2026-01

.PHONY: help install test run dashboard api gate gate-fail calibrate demo clean

help:
	@echo "make install     install dependencies (pip install -r requirements.txt)"
	@echo "make test        run the test suite"
	@echo "make run         run the demo evaluation (200 traces) -> $(OUT)"
	@echo "make dashboard   launch the Streamlit dashboard"
	@echo "make api         launch the FastAPI service on :8077"
	@echo "make gate        run the regression gate on the passing sample"
	@echo "make gate-fail   run the gate on the regressed sample (expects exit 1)"
	@echo "make calibrate   run judge calibration against the bundled labels"
	@echo "make demo        install + run + calibrate + gate (full walkthrough)"

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -q

run:
	$(PY) -m evalkit run --out $(OUT) --run-id $(RUN_ID) --with-judge

dashboard:
	$(PY) -m streamlit run dashboard/app.py

api:
	$(PY) -m uvicorn api.main:app --port 8077 --reload

gate:
	$(PY) -m evalkit.gate --baseline samples/baseline.json --current samples/current.json --config gate.yaml

gate-fail:
	$(PY) -m evalkit.gate --baseline samples/baseline.json --current samples/regressed.json --config gate.yaml; \
	test $$? -eq 1

calibrate:
	$(PY) -m evalkit.calibrate --labels samples/judge_calibration.jsonl --heuristic

demo: run calibrate gate

clean:
	rm -rf artifacts __pycache__ .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
