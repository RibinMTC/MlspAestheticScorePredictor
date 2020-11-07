#!/bin/bash

conda activate jupyter_env
pytest -v
conda activate jupyter_env27
pytest -v