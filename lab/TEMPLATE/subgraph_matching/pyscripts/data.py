"""데이터셋을 정의하는 모듈입니다.

`PyG`를 바탕으로 *Subgraph Matching Problem*에서 사용할 데이터셋을 정의합니다.
`./conf/data.yaml`에서 불러온 구성 정보를 바탕으로 처리합니다.

Example:

    dataset = SubgraphMatchingDataset(root="./data/")
    dataset.load("train")
"""

from utils.configure import Configure
from pyscripts.graph import Edge, Graph
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import Data, BaseData, DataEdgeAttr, DataTensorAttr, GlobalStorage
from os import PathLike
from itertools import permutations
from random import random, choice, randint, gauss
from typing import override
import torch
from torch.serialization import add_safe_globals

add_safe_globals([Data, BaseData, DataEdgeAttr, DataTensorAttr, GlobalStorage])

# configure 불러오기
conf = Configure(path="./conf/data.yaml")

# 데이터셋 정의
class SubgraphMatchingDataset(InMemoryDataset):
    """Subgraph Matching Problem 실험에 쓰일 데이터셋입니다.
    
    데이터셋은 `subgraph`, `train`, `validate`, `test`로 구성됩니다.
    `subgraph`는 데이터셋에서 공유하는 부분그래프를 의미합니다.
    `train`, `validate`, `test`는 각각 학습, 검증, 테스트 데이터입니다.
    데이터는 모두 초기화 시에 자동으로 생성됩니다.
    `.load(path)`를 사용하여 데이터를 불러올 수 있으며,
    `.get(idx)`를 사용하여 데이터를 참조할 수 있습니다.

    Example:

        dataset = SubgraphMatchingDataset(root="./data/")
        dataset.load("train")
    """

    __slots__ = ()

    def __init__(self, root: PathLike) -> None:
        """데이터셋을 초기화합니다.
        
        지정된 `root` 디렉토리 하위에 데이터셋을 만듭니다.
        데이터는 다음과 같은 구조로 저장됩니다.

        ```
        root
        ├───subgraph.pt         # 부분그래프 저장용 파일
        ├───train.pt            # 학습 데이터
        ├───validate.pt         # 검증 데이터
        ├───test.pt             # 테스트 데이터
        ├───pre_transform.pt    # (무시)
        └───pre_filter.pt       # (무시)

        ```

        Args:
            root: 데이터셋 디렉토리의 주소

        """
        super().__init__(root)

    # 이 데이터셋은 '다운로드'를 받는 게 아니고 새로 데이터를 생성하는 것이기 때문에
    # 별도로 'raw' 파일을 사용하지 않습니다.
    # 대신 바로 'process' 단계에서 데이터를 생성합니다.

    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return ['subgraph.pt', 'train.pt', 'validate.pt', 'test.pt']

    @property
    def processed_dir(self):
        return self.root
    
    # 데이터 생성 로직
    @override
    def process(self):
        """데이터를 생성합니다.
        
        원래 `InMemoryDataset`의 `process`함수는 데이터 가공 과정을 서술합니다.
        그러나 이번 실험의 목적과 설계 방식에 맞게, 데이터를 생성하는 로직을 서술합니다.
        데이터셋은 부분 그래프와, 데이터(train, validate, test)로 나뉩니다.
        둘 모두 *PyG*의 `Data` 형식으로 저장됩니다.
        """

        subgraph = _generate_subgraph(
            order=conf.num_subgraph_nodes,
            delta=conf.delta 
        )

        torch.save(self.collate([subgraph.to_data()]), self.processed_paths[0])

        full_data = {"train": [], "validate": [], "test": []}

        for i, key in enumerate(full_data.keys(), start=1):
            for _ in range(conf.num_data[key]):
                graph = _generate_supergraph(
                    subgraph=subgraph,
                    order=conf.num_graph_nodes,
                    delta=conf.delta
                )
                feature = torch.tensor([
                    [randint(0, 10)+gauss(sigma=conf.feature_std)]
                    for _ in graph.nodes
                ])
                y=torch.tensor([1 if node in graph.subnodes(subgraph) else -1 for node in graph.nodes])

                full_data[key].append(
                    Data(
                        x=feature,
                        edge_index=graph.edge_index,
                        y=y
                    )
                )
            torch.save(self.collate(full_data[key]), self.processed_paths[i])

# 여기서부터는 논문에서 소개된 데이터 생성 방식을 다룹니다.
# 논문의 설명에 의하면,
# 1. 부분 그래프로 사용할 그래프 S를 만든다.
# 2. S를 포함하는 더 큰 그래프 G를 만든다.
# 와 같이 데이터셋을 제작합니다.

def _generate_subgraph(order: int, delta: float) -> Graph:
    """그래프를 생성합니다.
    
    논문에 소개된 방식을 사용하여 (부분)그래프를 생성합니다.
    1. 가능한 모든 엣지에 대해 `delta`의 확률로 엣지을 연결합니다.
    2. 연결그래프가 될 때까지 하나씩 엣지을 추가합니다.

    Args:
        order: 그래프의 위수
        delta: 엣지의 연결 확률
    """
    edges: set[Edge] = set()
    non_edges: set[Edge] = set()

    for u, v in permutations(range(order), 2):
        if random() < delta:
            edges.add(Edge(u, v))
        else:
            non_edges.add(Edge(u, v))
    
    graph = Graph(order=order, edges=edges)

    while not graph.connected():
        edge = choice(non_edges)
        non_edges.remove(edge)
        graph.add(edge)

    return graph

def _generate_supergraph(subgraph: Graph, order: int, delta: float) -> Graph:
    """상위 그래프(supergraph)를 생성합니다.
    
    `subgraph`를 포함하되 위수가 `order`인 더 큰 그래프를 생성합니다.
    연결 확률(`delta`)에 따라 엣지는 무작위로 생성됩니다.

    Args:
        subgraph: 포함해야 하는 부분그래프
        order: 그래프의 위수
        delta: 엣지의 연결 확률
    """
    edges: set[Edge] = set(subgraph._edges)
    non_edges: set[Edge] = set()

    for u, v in permutations(range(order), 2):
        if Edge(u, v) in edges:
            continue

        if random() < delta:
            edges.add(Edge(u, v))
        else:
            non_edges.add(Edge(u, v))

    graph = Graph(order=order, edges=edges)

    while not graph.connected():
        edge = choice(non_edges)
        non_edges.remove(edge)
        graph.add(edge)
    
    return Graph(order=order, edges=edges)