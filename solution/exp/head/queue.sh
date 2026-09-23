#!/bin/zsh
# usage: queue.sh NAME [NAME...]   runs hcv.py sequentially (cv + bootstrap vs tx_v1f), one log per config
cd "$(dirname "$0")"
source ../../../.venv/bin/activate
export OMP_NUM_THREADS=4
for n in "$@"; do
  a=(${(s: :)n})
  OMP_NUM_THREADS=4 nice -n 15 python hcv.py ${a[@]} > "logs/${a[1]}.log" 2>&1
  echo "done ${a[1]}: $(grep -E '^t=0.5|cv \[' logs/${a[1]}.log | tr '\n' ' ')"
done
echo QUEUE_END
