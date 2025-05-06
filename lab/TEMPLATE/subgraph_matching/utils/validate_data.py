"""데이터셋을 검증합니다.

이 파일은 `db.sh`를 통해 실행되는 파일입니다.
데이터셋이 발견되거나 성공적으로 생성된 경우 실행됩니다.
`--data_dir`에 존재하는 데이터셋이 적절한 구조인지 검증합니다.
문제가 발생하는 경우 `data/log.txt`에 오류를 출력합니다.

검증하는 것들은 다음과 같습니다.
- `subgraph.pt`, `train.pt`, `validate.pt`, `test.pt`가 존재하는가?
- 각 데이터는 *PyG*의 `Data` 타입으로 불러와지는가?
- 데이터의 형식이 `conf/data.yaml`에 명시된 바를 따르는가?
- 각 데이터의 그래프들이 부분그래프를 포함하는가?
"""

from utils.configure import Configure
import argparse
import os.path as osp
import traceback
import logging
import sys
from pyscripts.graph import Graph, Edge
from pyscripts.data import SubgraphMatchingDataset
from torch_geometric.data.data import Data

# 구성 불러오기
conf = Configure("./conf/data.yaml")

# 경로 파싱
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir")
args = parser.parse_args()
root: str = args.data_dir

# 로그 설정
log_path = osp.join(root, "log.txt")
logging.basicConfig(filename=log_path, level=logging.ERROR)

# 데이터셋 검증 시도
try:
    dataset = SubgraphMatchingDataset(root=root)

    for filename in ["subgraph.pt", "train.pt", "validate.pt", "test.pt"]:
        path = osp.join(root, filename)

        # Step 1. 데이터 파일 존재 검증
        if not osp.exists(path):
            raise FileNotFoundError(f"데이터 파일이 존재하지 않습니다: {filename}")
        
        # Step 2. 데이터 파일 타입(Data) 검증
        dataset.load(path)
        if not isinstance(dataset._data, Data):
            raise TypeError(f"{filename}에 저장된 데이터의 타입이 'Data'가 아닌 {dataset._data.__class__.__name__} 입니다.")
        
        # Step 3. 데이터 내부 형식 검증
        if filename == "subgraph.pt":
            data: Data = dataset.get(0)
            if (
                data.num_nodes != conf.num_subgraph_nodes
                or data.edge_index is None
            ):
                raise ValueError(f"서브그래프가 'conf/data.yaml'의 형식을 따르지 않습니다: {data}")
            subgraph = Graph(order=data.num_nodes, edges=list(map(lambda x: Edge(*x), zip(*data.edge_index.tolist()))))
        else:
            for data in dataset:
                if (
                    data.num_nodes != conf.num_graph_nodes
                    or data.edge_index is None
                    or data.x is None
                    or data.y is None
                ):
                    raise ValueError(f"데이터 {filename}가 'conf/data.yaml'의 형식을 따르지 않습니다: {data}")

                # Step 4. 부분그래프 검증
                graph = Graph(order=data.num_nodes, edges=list(map(lambda x: Edge(*x), zip(*data.edge_index.tolist()))))
                if not subgraph in graph:
                    raise ValueError(f"데이터 {filename}에 부분그래프를 포함하지 않는 데이터가 발견되었습니다: {data}")
        
except Exception as e:
    logging.error("Exception occurred:\n%s", traceback.format_exc())
    sys.exit(1)