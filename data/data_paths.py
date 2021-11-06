#!/usr/bin/env python
# encoding: utf-8

from pathlib import Path

NN_DATAPATH = Path('data/models')
SKLEARN_DATAPATH = Path('data/ml_models')

NN_DATAPATH.mkdir(parents=True, exist_ok=True)
SKLEARN_DATAPATH.mkdir(parents=True, exist_ok=True)
