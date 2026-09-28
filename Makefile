# COVE: distribution-free false-accept control for embedding-based verification.
# All commands run from the repository root on one GPU (<= 24 GB).
PY ?= .venv/bin/python
SHELL := /bin/bash

.PHONY: setup data features smoke test pilot core breadth cohort adverse toy zoo pac oracle tables figures audit paper reproduce

setup:            ## create the environment from the lock file
	uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.lock

data:             ## download public datasets into the Hugging Face cache
	$(PY) scripts/download_data.py

features:         ## cache frozen embeddings (under 1 GPU-hour on a recent GPU)
	for ds in imagenet sop cub cars inshop gldv2mini; do $(PY) scripts/extract_features.py --dataset $$ds --encoders dinov2_b clip_b16 resnet50; done
	for ds in sop imagenet; do $(PY) scripts/extract_features.py --dataset $$ds --encoders dinov2_s dinov2_l clip_l14 siglip_b16 convnext_b; done
	$(PY) scripts/dump_meta.py

test smoke:       ## unit, validity and engine tests (< 1 minute)
	$(PY) -m pytest -q

pilot:            ## the Day-1 kill-gate pilot (random identity split, 100 draws)
	TAG=pilot2 scripts/run_pilot.sh 0 imagenet:dinov2_b:100,300 sop:dinov2_b:100,300,1000,3000

core:             ## Table 1 / Figures 1-2 (Tier A, 500 draws)
	scripts/run_tier.sh 0 tierA 500 500000 raw random imagenet:dinov2_b:10,30,100,300 imagenet:clip_b16:10,30,100,300 imagenet:resnet50:10,30,100,300 cub:dinov2_b:10,30,60 cub:clip_b16:10,30,60 cub:resnet50:10,30,60 cars:dinov2_b:10,30,60 cars:clip_b16:10,30,60 cars:resnet50:10,30,60 sop:dinov2_b:30,100,300,1000,3000,10000 sop:clip_b16:30,100,300,1000,3000,10000 sop:resnet50:30,100,300,1000,3000,10000

toy:
	$(PY) scripts/run_toy.py --reps 500 --ms 10 30 100 300 1000 3000

breadth cohort adverse:
	scripts/queue_after_tierA.sh 0 $@

pac:             ## PAC / identity-bootstrap tier on one GPU
	EXTRA=--with-bootstrap scripts/run_tier.sh 0 pac 200 500000 raw random imagenet:dinov2_b:30,100,300 imagenet:clip_b16:30,100,300 imagenet:resnet50:30,100,300 cub:dinov2_b:30,60 cub:clip_b16:30,60 cub:resnet50:30,60 sop:dinov2_b:30,100,300,1000 sop:clip_b16:30,100,300,1000 sop:resnet50:30,100,300,1000 cars:dinov2_b:30,60 cars:clip_b16:30,60 cars:resnet50:30,60

oracle:          ## population reference curves at COVE's exact level
	bash scripts/oracle_all.sh 0

tables figures:
	for t in tierA tierB tierC cohort adverse pac toy; do $(PY) tools/aggregate.py $$t; done
	$(PY) paper/build_evidence.py
	$(PY) tools/build_figures.py --main tierA --toy toy --out paper/figures

audit:            ## provenance and consistency gate; fails loudly
	$(PY) tools/audit.py

paper: tables figures
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

reproduce: setup test tables figures paper audit   ## rebuild the paper from shipped raw logs; see README
