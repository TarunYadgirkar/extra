#!/bin/zsh
# usage: run_folds.sh TAG [key=value ...]  -> trains folds 0..3 + all, then scores alone and stacked
cd "$(dirname $0)/../../.." && source .venv/bin/activate
export OMP_NUM_THREADS=4
TAG=$1; shift
for f in 0 1 2 3 all; do nice -n 15 python solution/exp/neural/train_nn.py $TAG $f "$@" || exit 1; done
nice -n 15 python solution/exp/neural/eval_nn.py $TAG --stack --base ${BASE:-v3+tx_v1} > solution/exp/neural/logs/${TAG}_eval.log 2>&1
