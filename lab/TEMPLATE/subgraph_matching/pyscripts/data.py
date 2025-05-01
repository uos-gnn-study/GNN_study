from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import to_networkx
from typing import override, Iterator, Literal
import yaml
import torch
import os.path as osp
from random import random, shuffle, randint, gauss
from itertools import combinations
from collections import defaultdict, deque
from dataclasses import dataclass
import networkx as nx

# Configure 가져오기
with open("./conf/data.yaml") as f:
    data_conf = yaml.load(f, yaml.FullLoader)

NUM_DATA                = data_conf["num_data"]                 # 데이터 수
NUM_SUBGRAPH_NODES      = data_conf["num_subgraph_nodes"]       # 서브그래프 노드수
NUM_GRAPH_NODES         = data_conf["num_graph_nodes"]          # 그래프 노드수
DELTA                   = data_conf["delta"]                    # 엣지 연결 확률
FEATURE_STD             = data_conf["feature_std"]

# Graph
@dataclass(slots=True)
class Graph:
    vertices: torch.Tensor
    edge_index: torch.Tensor

    @property
    def edges(self) -> Iterator[tuple[int, int]]:
        for edge in zip(*self.edge_index):
            yield edge

# Dataset
class SubgraphMatchingDataset(InMemoryDataset):
    def __init__(self, root: str) -> None:
        self.subgraph = self._generate_subgraph(node_count=NUM_SUBGRAPH_NODES, delta=DELTA)
        super().__init__(root)
        self.load(self.processed_paths[0])

    # Subgraph 생성
    def _generate_subgraph(self, node_count: int, delta: float) -> Graph:
        vertices = list(range(node_count))

        edges = deque()
        non_edges = deque()

        for u, v in combinations(vertices, 2):
            if random() <= delta:
                edges.append((u, v))
            else:
                non_edges.append((u, v))

        def _connected(vertices: list[int], edges: list[tuple[int]]) -> bool:
            graph = defaultdict(list)
            for u, v in edges:
                graph[u].append(v)
                graph[v].append(u)

            visited = set()
            queue = deque([vertices[0]])
            while queue:
                node = queue.popleft()
                if node not in visited:
                    visited.add(node)
                    queue.extend([neighbor for neighbor in graph[node] if neighbor not in visited])

            return len(visited) == len(vertices)

        shuffle(non_edges)
        while not _connected(vertices, edges):
            edges.append(non_edges.pop())

        return Graph(
            vertices=torch.tensor(vertices),
            edge_index=torch.tensor(list(zip(*edges)))
        )

    # Graph 생성
    def _generate_graph(self, node_count: int, delta: int, subgraph: Graph) -> Graph:
        vertices = list(range(node_count))

        edges = deque(subgraph.edges)
        non_edges = deque()

        for u, v in combinations(vertices, 2):
            if (u, v) in edges:
                continue

            if random() <= delta:
                edges.append((u, v))
            else:
                non_edges.append((u, v))

        return Graph(
            vertices=torch.tensor(vertices),
            edge_index=torch.tensor(list(zip(*edges)))
        )

    # subgraph와 동형인 graph에 속한 노드 찾기
    def _node_in_subgraph(self, subgraph: Graph, graph: Graph) -> list[bool]:
        subgraph_nx = to_networkx(Data(x=subgraph.vertices, edge_index=subgraph.edge_index))
        graph_nx = to_networkx(Data(x=graph.vertices, edge_index=graph.edge_index))

        k = subgraph.vertices.size(dim=0)
        result = [0] * graph.vertices.size(dim=0)

        for v in graph.vertices:
            others = [u for u in graph.vertices if u != v]
            if result[v]:
                continue

            for subset in combinations(others, k-1):
                candidate = list(subset) + [v]
                candidate_subgraph = graph_nx.subgraph(candidate)
    
                if nx.is_isomorphic(candidate_subgraph, subgraph_nx):
                    for n in candidate:
                        result[n] = 1
                        break

        return result
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'validate.pt', 'test.pt']

    @override
    @property
    def processed_dir(self) -> str:
        return self.root

    # ${data_dir}에 그래프 데이터 생성
    def process(self) -> None:
        data_list = []

        num_train_data = NUM_DATA["train"]
        num_validate_data = NUM_DATA["validate"]
        num_test_data = NUM_DATA["test"]

        for i in range(num_train_data + num_validate_data + num_test_data):
            graph = self._generate_graph(node_count=NUM_GRAPH_NODES, delta=DELTA, subgraph=self.subgraph)
            # 첨언
            # 논문을 열심히 읽어봤는데, 딱히 feature(label)를 주는 방법에 대한 서술을 발견하지 못했습니다.
            # 전 초깃값(label)이 의미가 없더라도, 메세지 파싱을 통해 구조적 특성을 반영하다보면 해결된다는 주장으로 해석했습니다.
            # 그래서 feature를 줄 때, 특별히 정보를 담지 않고, 무작위로 값을 넣도록 했습니다.
            # ** 만약에 실험 결과가 좋지 않게 나온다면 이것 때문일 수 있습니다 **
            feature = torch.Tensor([
                [randint(0, 10)+gauss(sigma=FEATURE_STD)]
                for _ in graph.vertices
            ])
            y=torch.tensor(self._node_in_subgraph(self.subgraph, graph))
            print(feature.shape, graph.edge_index.shape, y.shape)
            
            data_list.append(Data(x=feature, edge_index=graph.edge_index, y=y))

        torch.save(self.collate(data_list[:num_train_data]), self.processed_paths[0])
        torch.save(self.collate(data_list[num_train_data:num_train_data+num_validate_data]), self.processed_paths[1])
        torch.save(self.collate(data_list[num_train_data+num_validate_data:]), self.processed_paths[2])

    def set_type(self, type=Literal['train', 'validate', 'test']) -> None:
        self.load(self.processed_paths[['train', 'validate', 'test'].index(type)])