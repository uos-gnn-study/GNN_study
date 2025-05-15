"""데이터셋을 준비합니다.

이 파일은 `db.sh`를 통해 실행되는 파일입니다.
지정된 경로에 데이터셋이 존재하지 않는 경우에 실행됩니다.
`--data_dir`에 데이터셋을 생성하는 역할을 합니다.
문제가 발생하는 경우 `data/log.txt`에 오류를 출력합니다.
"""


from pyscripts.data import SubgraphMatchingDataset
import argparse
import os.path as osp
import sys
import traceback
import logging

# 경로 파싱
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir")
args = parser.parse_args()
root: str = args.data_dir

# 로그 설정
log_path = osp.join(root, "log.txt")
logging.basicConfig(filename=log_path, level=logging.ERROR)

# 데이터셋 초기화
try:
    dataset = SubgraphMatchingDataset(root=root)
except Exception as e:
    logging.error("Exception occurred:\n%s", traceback.format_exc())
    sys.exit(1)