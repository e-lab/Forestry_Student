#!/usr/bin/env bash

export HOME=$PWD/src
export TMP=$PWD/tmp

mkdir -p $HOME
mkdir -p $TMP

cd src 
streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.enableCORS false