@echo off
title RareVision AI - Streamlit Dashboard
echo Starting RareVision AI...
start "" "http://localhost:8501"
python -m streamlit run app.py --server.port 8501
pause

