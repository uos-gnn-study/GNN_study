#!/bin/bash

# setup.sh가 있는 위치
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo $SCRIPT_DIR

# 복사할 항목 (setup.sh 있는 디렉토리 기준)
FILES_TO_COPY=("conf" "pyscripts" "utils" "db.sh")

echo "🔧 Subgraph Matching 실험 파일을 현재 폴더($(pwd))로 복사합니다."

for item in "${FILES_TO_COPY[@]}"; do
  SRC="$SCRIPT_DIR/$item"
  if [ -e "$SRC" ]; then
    echo "📁 복사 중: $item"
    cp -r "$SRC" .
  else
    echo "⚠️ $item 가 $SCRIPT_DIR 에서 발견되지 않았습니다. 건너뜁니다."
  fi
done

echo "✅ 복사 완료! 이제 실험을 시작할 수 있습니다."