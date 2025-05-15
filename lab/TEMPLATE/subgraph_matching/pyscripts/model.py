"""GNN 모델을 정의하는 모듈입니다.


"""

import torch.nn as nn
import torch
from torch_geometric.utils import degree, to_undirected
from torch_scatter import scatter_add

# ----------테스트 버전입니다---------- #
class LinearGNN(nn.Module):
    """Linear GNN을 정의합니다.
    
    """

    def __init__(self, *, mu: float, state_dimension: int) -> None:
        super().__init__()
        self.mu = mu
        self.state_dimension = state_dimension
        self.transition_network = nn.Sequential(
            nn.Linear(20, 50),
            nn.Sigmoid(),
            nn.Linear(50, 40),
            nn.Sigmoid(),
            nn.Linear(40, 40),
            nn.Sigmoid(),
            nn.Linear(40, 10*state_dimension*state_dimension),
            nn.Sigmoid()
        )
        self.forcing_network = nn.Sequential(
            nn.Linear(10, 30),
            nn.Sigmoid(),
            nn.Linear(30, 50),
            nn.Sigmoid(),
            nn.Linear(50, 40),
            nn.Sigmoid(),
            nn.Linear(40, 10*state_dimension),
            nn.Sigmoid()
        )
        self.local_output_function = nn.Sequential(
            nn.Linear(10*state_dimension, 50),
            nn.Sigmoid(),
            nn.Linear(50, 40),
            nn.Sigmoid(),
            nn.Linear(40, 30),
            nn.Sigmoid(),
            nn.Linear(30, 10),
            nn.Sigmoid(),
        )
        self.states = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        self.states = torch.randn((num_nodes, self.state_dimension, 1))
        
        u_deg = degree(edge_index.flatten(), num_nodes=num_nodes)
        multiplier = self.mu / (self.state_dimension * u_deg)
        A = (
            multiplier.reshape(-1, 1, 1) * torch.reshape(
                self.transition_network(x.repeat(1, 2).flatten()),
                (10, self.state_dimension, self.state_dimension)
            )
        )

        b = self.forcing_network(x.flatten()).reshape(10, self.state_dimension, 1)
        new = torch.bmm(A, self.states) + b

        while torch.mean(new - self.states) > 0.5:
            self.states = new
            new = torch.bmm(A, self.states) + b

        result = self.local_output_function(self.states.flatten())
        
        return result
