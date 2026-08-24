@echo off
cd /d "C:\Users\hp\Downloads\kuas"
python api.py > flask.log 2>&1
echo Flask API started on port 5000
timeout 2
echo API should be running now