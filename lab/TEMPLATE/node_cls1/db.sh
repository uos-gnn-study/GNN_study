#!/bin/bash
# db.sh: Prepare dataset for node classification task (e.g., subgraph matching)
# Generates torch .pt files including edge_index, node_feat, node_label, and optionally masks.
# Verifies data structure after generation.

set -euo pipefail

# ------------------ 설정 ------------------
# 기본값
with_mask=true
data_name=subgraph_matching
data_dir=./data/${data_name}

# 경로 설정
. ./path.sh || echo "No path.sh found. Continue without it."

# ------------------ 생성 ------------------
if [ -f "${data_dir}/edge_index.pt" ]; then
    echo "[db.sh] Data found in ${data_dir}."
else
    echo "[db.sh] Data not found in ${data_dir}. Creating dataset at ${data_dir} ..."
    mkdir -p ${data_dir}

    python utils/data/prepare_data.py \
        --output_dir "${data_dir}" \
        --with_mask "${with_mask}"
    
    echo "[db.sh] Data generation completed."
fi

# ------------------ 검증 ------------------
echo "[db.sh] Verifying dataset structure..."

required_files=(
    "edge_index.pt"
    "node_feat.pt"
    "node_label.pt"
)

if $with_mask; then
    required_files+=("train_mask.pt" "val_mask.pt" "test_mask.pt")
fi

# 파일 존재 확인
for file in "$required_files[@]"; do
    if [ ! -f "${data_dir}/${file}" ]; then
    echo "[db.sh] ERROR: Missing required file: ${data_dir}/${file}"
    exit 1
    fi
done