#!/bin/bash
# Double-click to open RE-USE Lite in your browser (needs a Chromium browser such as Brave or Chrome).
cd "$(dirname "$0")" || exit 1
( sleep 1; open "http://localhost:8737/" ) &
exec python3 serve.py 8737
