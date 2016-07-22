#!/usr/bin/env sh

set -o errexit

export PYTHONPATH=$PYTHONPATH:"$(pwd)/../.."
echo $PYTHONPATH

#python preprocessing/__main__.py
luajit featureextraction/main.lua -outDir /home/alberto/development/features -data /home/alberto/development/aligned
python classification/__main__.py train