ROOT := ../..

.PHONY: install run cycle cycle-dry cycle-smoke test check

install:
	python -m pip install -e $(ROOT)
	python -m pip install -r requirements.txt

run:
	cd $(ROOT) && streamlit run applications/juricratic_research_radar/app.py

cycle:
	cd $(ROOT) && python applications/juricratic_research_radar/scripts/run_research_cycle.py

cycle-dry:
	cd $(ROOT) && python applications/juricratic_research_radar/scripts/run_research_cycle.py --dry-run --max-batches-per-phase 1 --cooldown-seconds 0

cycle-smoke:
	cd $(ROOT) && python applications/juricratic_research_radar/scripts/run_research_cycle.py --smoke

test:
	PYTHONPATH=. pytest -q tests

check:
	python -m compileall -q juricratic_radar app.py
	PYTHONPATH=. pytest -q tests
