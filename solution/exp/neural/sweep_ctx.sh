#!/bin/zsh
# usage: sweep_ctx.sh FILE FOLDS   (FILE lines: TAG cfg...; FOLDS e.g. "0" or "0 1 2 3 all")
cd "$(dirname $0)/../../.." && source .venv/bin/activate
export OMP_NUM_THREADS=4
while read -r TAG ARGS; do
  [ -z "$TAG" ] && continue
  for f in ${=2}; do nice -n 15 python solution/exp/neural/train_nn.py $TAG $f ctx=1 resid=1 ${=ARGS}; done
  if [ "$2" = "0" ]; then F=0; else F=; fi
  nice -n 15 python solution/exp/neural/eval_ctx.py $TAG $F >> solution/exp/neural/logs/sweep_ctx.txt 2>&1
done < $1
