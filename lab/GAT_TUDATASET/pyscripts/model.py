import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import to_dense_adj

class GATLayer(nn.Module):
	def __init__(
		self,
		*, 
		num_nodes: int, 
		num_heads: int, 
		in_channels: int, 
		out_channels: int,
		leakyrelu_slope: float = 0.2
	) -> None:
		super().__init__()
		self.num_nodes = num_nodes
		self.num_heads = num_heads
		self.in_channels = in_channels
		self.out_channels = out_channels

		self.linear_weight = nn.Parameter(torch.randn(num_heads, in_channels, out_channels))
		self.query_weight = nn.Parameter(torch.randn(num_heads, 1, out_channels))
		self.key_weight = nn.Parameter(torch.randn(num_heads, 1, out_channels))
		self.leakyrelu = nn.LeakyReLU(negative_slope=leakyrelu_slope)
		self.softmax = nn.Softmax(dim=2)
			
	def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
		linear_transformed_input = torch.matmul(
				self.linear_weight,
				torch.transpose(torch.unsqueeze(x, dim=0), 1, 2)
		)
		
		query_tensor = torch.matmul(self.query_weight, linear_transformed_input)
		key_tensor = torch.matmul(self.key_weight, linear_transformed_input)
		value_tensor = torch.transpose(linear_transformed_input, 1, 2) # 결과가 같음
		
		attention_coefficient = self.leakyrelu(
			torch.matmul(
					query_tensor,
					torch.transpose(key_tensor, 1, 2)
			)    # self.leakyrelu = nn.LeakyReLU(negative_slope=0.2) <- 논문에서 0.2로 함
		)
		
		adjacency_matrix = to_dense_adj(edge_index=edge_index, max_num_nodes=self.num_nodes)
		masked_attention_coefficient = self.softmax(
				attention_coefficient.masked_fill(adjacency_matrix==0, value=-1e9)
		)    # self.softmax = nn.Softmax(dim=2)
		
		output = torch.reshape(
				torch.matmul(
						masked_attention_coefficient,
						value_tensor
				).permute(1, 0, 2),
				shape=(self.num_nodes, self.num_heads*self.out_channels)
		)
		
		return output

class GATLastLayer(GATLayer):
	def __init__(
		self,
		*, 
		num_nodes: int, 
		num_heads: int, 
		in_channels: int, 
		out_channels: int,
		leakyrelu_slope: float = 0.2
	) -> None:
		super().__init__(
			num_nodes=num_nodes, 
			num_heads=num_heads, 
			in_channels=in_channels,
			out_channels=out_channels, 
			leakyrelu_slope=leakyrelu_slope
		)

	def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
		linear_transformed_input = torch.matmul(
			self.linear_weight,
			torch.transpose(torch.unsqueeze(x, dim=0), 1, 2)
		)
		
		query_tensor = torch.matmul(self.query_weight, linear_transformed_input)
		key_tensor = torch.matmul(self.key_weight, linear_transformed_input)
		value_tensor = torch.transpose(linear_transformed_input, 1, 2) # 결과가 같음
		
		attention_coefficient = self.leakyrelu(
			torch.matmul(
					query_tensor,
					torch.transpose(key_tensor, 1, 2)
			)    # self.leakyrelu = nn.LeakyReLU(negative_slope=0.2) <- 논문에서 0.2로 함
		)
		
		adjacency_matrix = to_dense_adj(edge_index=edge_index, max_num_nodes=self.num_nodes)
		masked_attention_coefficient = self.softmax(
				attention_coefficient.masked_fill(adjacency_matrix==0, value=-1e9)
		)    # self.softmax = nn.Softmax(dim=2)
			
		output = torch.mean(
			torch.matmul(
					masked_attention_coefficient,
					value_tensor
			),
			dim=0
		)
		
		return output
	
class GATModel(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_nodes: int,
        num_heads: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        leakyrelu_slope: float = 0.2
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        for i in range(num_layers):
            input_dim = in_channels if i == 0 else hidden_channels * num_heads
            output_dim = out_channels if i == num_layers - 1 else hidden_channels

            layer_cls = GATLastLayer if i == num_layers - 1 else GATLayer
            self.layers.append(
                layer_cls(
                    num_nodes=num_nodes,
                    num_heads=num_heads,
                    in_channels=input_dim,
                    out_channels=output_dim,
                    leakyrelu_slope=leakyrelu_slope
                )
            )

            if i != num_layers - 1:
                self.activations.append(nn.ELU(alpha=1.0))
            else:
                self.activations.append(nn.Sigmoid())

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for layer, act in zip(self.layers, self.activations):
            x = layer(x, edge_index)
            x = act(x)
        return x