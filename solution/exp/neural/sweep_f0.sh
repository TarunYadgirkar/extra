#!/bin/zsh
# fold-0 sweep: each line = TAG cfg...; runs sequentially
cd "$(dirname $0)/../../.." && source .venv/bin/activate
export OMP_NUM_THREADS=4
while read -r TAG ARGS; do
  [ -z "$TAG" ] && continue
  nice -n 15 python solution/exp/neural/train_nn.py $TAG 0 eval_every=500 ${=ARGS}
  nice -n 15 python solution/exp/neural/fold_score.py $TAG 0 >> solution/exp/neural/logs/sweep_f0.txt 2>&1
done < $1
