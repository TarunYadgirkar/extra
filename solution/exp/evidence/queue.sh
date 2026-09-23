#!/bin/zsh
# usage: queue.sh "NAME [--val] [--save]" ...   waits for build_ev.py, then runs evcv.py sequentially
cd "$(dirname "$0")"
source ../../../.venv/bin/activate
while pgrep -f build_ev.py > /dev/null; do sleep 10; done
for n in "$@"; do
  a=(${(s: :)n})
  OMP_NUM_THREADS=4 nice -n 15 python evcv.py ${a[@]} > "logs/${a[1]}.log" 2>&1
  echo "done ${a[1]}: $(grep -E '^t=0.5|cv \[|val \[|Error' logs/${a[1]}.log | tr '\n' ' ')"
done
echo QUEUE_END
