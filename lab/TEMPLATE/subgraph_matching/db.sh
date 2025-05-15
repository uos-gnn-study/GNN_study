#!/bin/bash

set -uo pipefail

export PYTHONPATH='.'

# ------------------ 생성 ------------------
if [ -d ${data_dir} ] && find ${data_dir} -name "*.pt" | grep -q . ; then
    echo "[db.sh] 데이터 디렉토리가 발견되었습니다."
else
    echo "[db.sh] 데이터 디렉토리가 발견되지 않았습니다. 데이터를 새로 생성합니다."
    mkdir -p ${data_dir}
    touch ${data_dir}/log

    args="--data_dir ${data_dir}"

    python utils/prepare_data.py ${args}
    status=$?

    if [ $status -ne 0 ]; then
    echo "[db.sh] 데이터 생성 과정에서 문제가 발생하였습니다."
    exit 1
    else
    echo "[db.sh] 데이터 생성이 완료되었습니다."
    fi
fi

# ------------------ 검증 ------------------
echo "[db.sh] 데이터셋을 검증합니다."

args="--data_dir ${data_dir}"

python utils/validate_data.py ${args}
status=$?

if [ $status -ne 0 ]; then
    echo "[db.sh] 데이터 검증 과정에서 문제가 발생하였습니다."
    exit 1
else
    echo "[db.sh] 데이터 검증이 완료되었습니다."
fi