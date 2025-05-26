from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from pyscripts.configure import Configure
from pyscripts.model import GATModel
import torch
import torch.nn.functional as F


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_conf = Configure("./conf/train.yaml")
    model_conf = Configure("./conf/model.yaml")

    dataset = TUDataset(root="./lab/GAT_TUDATASET/data/", name="PROTEINS").to(
        device=device
    )
    model = GATModel()

    loader = DataLoader(dataset=dataset, batch_size=train_conf.batch_size)
    optim_cls = getattr(torch.optim, train_conf.optimizer, torch.optim.Adam)
    optimizer = optim_cls(model.parameters())

    for batch in loader:
        optimizer.zero_grad()
        batch = batch.to(device=device)
        out = model(batch.x, batch.edge_index)

        y = batch.y[: batch.batch_size]
        out = out[: batch.batch_size]

        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

    return


if __name__ == "__main__":
    main()
