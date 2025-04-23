#!/bin/bash

# Run Flask app with conda environment
export FLASK_APP=app.py
/Users/jakubsladek/miniconda3/envs/chatlumos/bin/python -m flask run "$@"
