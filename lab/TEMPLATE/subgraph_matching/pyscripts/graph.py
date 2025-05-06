"""여러 유틸리티를 지원하는 그래프와 엣지를 정의하는 모듈입니다.

*Subgraph Matching Problem*은 데이터를 생성하기 위해 여러 도구를 필요로 합니다.
하지만 이를 `networkx.Graph`나 `torch_geometric.data.Data`로 구현하기엔 번잡합니다.
가독성과 유지 보수성을 확보하기 위해 이러한 모듈이 제작되었습니다.

Example:

    graph = Graph(order=3, edges=Edge(0, 2))
    graph.edge_index        # torch.Tensor([[0], [2]])
"""

from __future__ import annotations
from collections.abc import Collection, Mapping
from ctypes import c_int
from typing import Iterator, overload
from itertools import permutations
import torch
from torch_geometric.data import Data

# 노드 정의
type Node = int

# 엣지 정의
class Edge(Collection):
    """엣지를 선언하는 클래스입니다.
    
    무방향 그래프에서 엣지는 가변적이지 않고 순서가 없으며 크기가 정해진 구조체입니다.
    하지만 실제로 이러한 타입은 *Python*에서 제공하지 않습니다.
    속도와 안전성을 위해서는 C-수준에서 엣지를 구현하는 클래스를 선언하는 것이 좋습니다.
    `Edge`클래스는 C의 배열을 기반으로 빠르게 접근 가능하면서도 가볍고 안전하게 제작되었습니다.

    Example:
        
        Edge(2, 3)
        assert 2 in Edge        # True
    """

    __slots__ = ("_store")

    def __init__(self, *nodes: Node):
        """엣지를 생성합니다.
        
        입력된 두 음이 아닌 정수로 구성된 Edge를 생성합니다.
        두 음이 아닌 정수가 입력되지 않은 경우에는 오류를 발생시킵니다.

        Args:
            nodes: 엣지를 구성하는 두 노드입니다.

        Raises:
            TypeError: 두 정수가 입력되지 않은 경우에 발생하는 오류
            ValueError: 음의 정수가 입력된 경우에 발생하는 오류
        """

        # 오류 검정
        if (
            len(nodes) != 2
            or any(node for node in nodes if not isinstance(node, int))
        ):
            raise TypeError(f"정수 쌍이 아닌 {nodes}가 입력되었습니다.")
        if any(node for node in nodes if node < 0):
            raise ValueError(f"음의 정수가 입력되었습니다: {nodes}")
        
        # 엣지 생성 (w. ARRAY)
        nodes = tuple(sorted(nodes))
        self._store = (c_int * 2)(*nodes)
    
    def __contains__(self, node: Node) -> bool:
        """노드가 그래프에 속하는지 확인합니다."""
        return node in self._store
    
    def __eq__(self, other: Edge) -> bool:
        """두 엣지가 같은지 확인합니다."""
        return (
            isinstance(other, Edge)
            and set(self._store) == set(other._store)
        )
    
    def __iter__(self) -> Iterator[Node]:
        return iter(self._store)
    
    def __repr__(self) -> str:
        return f"Edge({self._store[0]}, {self._store[1]})"
    
    def __len__(self) -> int:
        return 2
    
    def __hash__(self) -> int:
        return hash(tuple(self._store))

# 그래프 정의
class Graph:
    """그래프 데이터를 저장하는 클래스입니다.

    *Subgraph Matching Problem*에서 그래프 데이터는 상당히 간단한 구조를 가집니다.
    노드의 수와, 그 연결 관계만 저장하면 그 외의 것들은 그다지 필요한 정보가 아닙니다.
    따라서 복잡한 외부 라이브러리의 클래스보다, 효율적인 클래스를 따로 선언하는 것이 좋습니다.
    `Graph`클래스는 그래프를 가능한 간소하게 표현하며, 이 태스크에 필요한 다양한 동작을 지원합니다.

    Example:

        graph = Graph(
            order=5,
            edges={{0, 5}, {1, 2}, {3, 4}}
        )
        graph.nodes     # Iterable: 0, 1, 2, 3, 4
        graph.edges     # Iterable: {0, 5}, {1, 2}, {3, 4}
    """

    __slots__ = ("_order", "_edges")

    def __init__(self, order: int, edges: Collection[Edge]) -> None:
        """그래프 인스턴스를 생성합니다.

        입력된 그래프의 정보를 바탕으로 그래프를 생성합니다.
        `edges`는 `Edge`의 모음일 것으로 기대합니다.
        `edges`의 모든 음이 아닌 정수는 `order`보다 작을 것으로 기대합니다.
        그 외의 경우 오류를 발생시킵니다.

        Args:
            order: 그래프에 포함된 노드의 수
            edges: 그래프에 포함된 엣지들

        Raises:
            TypeError: `order`에 `int`가 아닌 값이 입력되면 발생하는 오류
            ValueError: `order`가 0보다 작을 때 발생하는 오류
            TypeError: `edges`에 `Edge` 타입이 아닌 원소가 존재하면 발생하는 오류
            ValueError: `edges`에 `order`보다 큰 정수가 존재할 때 발생하는 오류
        """

        if not isinstance(order, int):
            raise TypeError(f"order에 `int`가 아닌 값이 입력되었습니다: {order}")
        
        if order < 0:
            raise ValueError(f"order에 음수가 입력되었습니다: {order}")

        if len(edges) != 0:
            if any(not isinstance(edge, Edge) for edge in edges):
                raise TypeError(f"Edge 타입이 아닌 edge가 존재합니다: {edges}")

            if any(node >= order for edge in edges for node in edge):
                raise ValueError(f"order보다 큰 노드가 존재합니다")
        
        self._order = order
        self._edges = edges

    @property
    def nodes(self) -> Iterator[Node]:
        """그래프의 노드들을 반환하는 이터레이터"""
        return iter(range(self._order))

    @property
    def order(self) -> int:
        """그래프의 위수(노드의 수)"""
        return self._order
    
    @property
    def edges(self) -> Iterator[Edge]:
        """그래프의 엣지들을 반환하는 이터레이터"""
        return iter(self._edges)
    
    @property
    def edge_index(self) -> torch.Tensor:
        """그래프의 `edge_index`(간선 행렬)"""
        return torch.tensor(list(zip(*self._edges)))
    
    def neighbors(self) -> Mapping[Node, set[Node]]:
        """자신을 포함해 각 노드에서 이웃들로의 매핑을 반환하는 함수"""
        neighbor_map = {node: set() for node in self.nodes}
        for edge in self.edges:
            u, v = edge
            neighbor_map[u].add(v)
            neighbor_map[v].add(u)
        return neighbor_map
    
    @overload
    def __contains__(self, node: int) -> bool:
        """`node`가 그래프에 속하는지 확인합니다"""

    @overload
    def __contains__(self, edge: Edge) -> bool:
        """`edge`가 그래프에 속하는지 확인합니다"""

    @overload
    def __contains__(self, subgraph: Graph) -> bool:
        """`subgraph`가 그래프에 속하는지 확인합니다"""

    def __contains__(self, arg: int | Edge | Graph) -> bool:
        # 1. 노드인 경우
        if isinstance(arg, int):
            return arg < self.order
        
        # 2. 엣지인 경우
        elif isinstance(arg, Edge):
            return arg in self._edges
        
        # 3. 그래프인 경우(부분그래프 확인)
        elif isinstance(arg, Graph):
            return bool(len(self.subnodes(arg)))
                
        # 4. 그 외의 경우(무조건 거짓)
        return False
        
    def connected(self) -> bool:
        """그래프의 연결성을 검증하는 함수"""
        visited = set()
        neighbors = self.neighbors()
        opened: list[Node] = [0]

        while opened:
            node = opened.pop()

            if node in visited:
                continue

            opened.extend(neighbors[node])
            visited.add(node)
        
        return len(visited) == self.order

    def subnodes(self, subgraph: Graph) -> Collection[Node]:
        """`subgraph`와 동형이며, 그래프에 속하는 부분그래프의 노드들을 반환하는 함수"""
        # cf. 이 문제는 "Subgraph Isomorphism Problem"라 불리는 NP 문제입니다.
        #     이 코드의 최적화에는 많은 시간을 쏟을 필요가 없습니다.
        if not isinstance(subgraph, Graph):
            raise TypeError(f"`subgraph`에 인자로 Graph가 아닌 객체가 전달되었습니다: {subgraph}")
    
        subgraph_adjacency = subgraph.neighbors()
        graph_adjacency = self.neighbors()

        for transform in permutations(self.nodes, subgraph.order):
            valid = True

            for node, neighbors in subgraph_adjacency.items():
                node = transform[node]
                neighbors = {transform[neighbor] for neighbor in neighbors}
                if not neighbors.issubset(graph_adjacency[node]):
                    valid = False
                    break
            if valid:
                return [transform[node] for node in subgraph.nodes]
            
        return []
    
    def to_data(self) -> Data:
        """*PyG*의 `Data` 형식으로 변환하는 함수"""
        return Data(num_nodes=self.order, edge_index=self.edge_index)
    
    @overload
    def add(self, num_nodes: int) -> None:
        """`num_nodes`만큼 노드를 추가합니다."""

    @overload
    def add(self, edge: Edge) -> None:
        """엣지(`edge`)를 추가합니다"""

    def add(self, arg: int | Edge) -> None:
        if isinstance(arg, int):
            if arg <= 0:
                raise ValueError(f"자연수가 아닌 개수의 노드를 추가하려 했습니다: num_nodes={arg}")
            self._order += arg
        elif isinstance(arg, Edge):
            if any(node >= self.order for node in arg):
                raise ValueError(f"존재하지 않는 노드와 간선을 연결하려 시도했습니다: edge={arg}")
            self._edges = set(self._edges)
            self._edges.add(arg)
        else:
            raise TypeError(f"잘못된 타입의 인자가 입력되었습니다: {arg.__class__.__name__}")