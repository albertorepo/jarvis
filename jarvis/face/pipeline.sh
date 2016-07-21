#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:/Users/albertocastano/development/jarvis

python preprocessing/__main__.py
luajit featureextraction/main.lua
python classification/__main__.py train