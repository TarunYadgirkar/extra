#!/bin/zsh
source ../../../.venv/bin/activate
while pgrep -f "gen_maps.py val" >/dev/null; do sleep 10; done
export OMP_NUM_THREADS=4
nice -n 15 python gen_maps.py fit > logs/fit.log 2>&1 && nice -n 15 python gen_maps.py train > logs/maps_train.log 2>&1
echo CHAIN_DONE $? >> logs/maps_train.log
