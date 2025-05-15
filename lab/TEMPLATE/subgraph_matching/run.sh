#!/bin/bash

set -uo pipefail

# ------------------ 설정 ------------------
data_name=subgraph_matching
data_dir=data/${data_name}/
experiment_dir=exp_ex1

# ------------------ 실행 ------------------
echo "[run.sh] 프로젝트를 실행합니다"

# 데이터 생성 및 검증
echo "[run.sh] Step 1. 데이터 생성 및 검증"
source db.sh
status=$?

if [ $status -ne 0 ]; then
    echo "[run.sh] 오류가 발생하였으므로 프로젝트를 종료합니다."
    exit 1
fi

# 모델 학습 실행
echo "[run.sh] Step 2. 모델 학습"
mkdir -p ${experiment_dir}

args="--data_dir ${data_dir} --experiment_dir ${experiment_dir}"

python pyscripts/execute.py ${args}
status=$?

if [ $status -ne 0 ]; then
    echo "[run.sh] 오류가 발생하였으므로 프로젝트를 종료합니다."
    exit 1
fi