#!/bin/bash

set -euo pipefail

# ------------------ 설정 ------------------
# 기본값
data_name=subgraph_matching
data_dir=data/${data_name}/

# ------------------ 생성 ------------------
if [ -d ${data_dir} ]; then
    echo "[db.sh] 데이터 디렉토리가 발견되었습니다."
else
    echo "[db.sh] 데이터 디렉토리가 발견되지 않았습니다. 데이터를 새로 생성합니다."
    mkdir -p ${data_dir}

    args="--output_dir ${data_dir}"

    export PYTHONPATH='.'
    python utils/prepare_data.py ${args}
    echo "[db.sh] 데이터 생성이 완료되었습니다."
fi

# ------------------ 검증 ------------------
echo "[db.sh] 데이터셋 구조를 검증합니다."

required_files=(
    "train.pt"
    "validate.pt"
    "test.pt"
)

for file in "$required_files"; do
    if [ ! -f "${data_dir}/${file}" ]; then
        echo "[db.sh] ERROR: 필요한 파일이 존재하지 않습니다: ${data_dir}${file}"
        exit 1
    fi
done

# TODO: 검증 완성하기
