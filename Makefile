#  Makefile 
# Top-level helper for one-shot replication

PY = python -m

.PHONY: reproduce clean sim curate plots tables

# remove outputs/ and figures/, keep previews tracked in git
clean:
	@echo "[make] cleaning artefacts …"
	rm -rf outputs/* figures/* 2>/dev/null || true

# pipeline steps – split so they can be run individually
sim:
	$(PY) sim_runner --config scenarios.yaml --out outputs/simulations.parquet

curate:
	$(PY) curation

plots:
	$(PY) visualise

tables:
	$(PY) table_exporter

# the one-liner requested in the plan (H-1)
reproduce: clean sim curate plots tables
	@echo "[make] ✅ full reproduction finished – artefacts under outputs/, figures/ & tables/"
# ──────────────────────────────────────────────────────────────
