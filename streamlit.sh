#!/usr/bin/env bash

export HOME=$PWD/src
export TMP=$PWD/tmp

export GOOGLE_API_KEY=AIzaSyDKxAadUfBZ9oAMDlRjRe0jlp3N0oZKqvg
export GOOGLE_CSE_ID=57d010b1a25ce48c0

mkdir -p $HOME
mkdir -p $TMP

cd src 
streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.enableCORS false