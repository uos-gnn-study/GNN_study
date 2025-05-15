from pyscripts.model import LinearGNN
from utils.configure import Configure
from pyscripts.data import SubgraphMatchingDataset
import argparse
import logging
import os.path as osp
import traceback
import sys
from torch_geometric.data import Data
import torch
from tqdm import tqdm

# conf = Configure("conf/model.yaml")

# 경로 파싱
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir")
parser.add_argument("--experiment_dir")
args = parser.parse_args()
data_root: str = args.data_dir
exp_root: str = args.experiment_dir

# 로그 설정
log_path = osp.join(exp_root, "log.txt")
logging.basicConfig(filename=log_path, level=logging.ERROR)

# 평가 함수
def evaluate(model: LinearGNN, dataset: SubgraphMatchingDataset):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataset:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            pred = out.round()
            correct += (pred == data.y.float()).sum().item()
            total += data.y.size(0)
    return correct / total if total > 0 else 0.0

try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_raw = SubgraphMatchingDataset(root=data_root)
    dataset: dict[str, SubgraphMatchingDataset] = {}
    # loader: dict[str, DataLoader] = {}
    for name in ("train.pt", "validate.pt", "test.pt"):
        data_raw.load(osp.join(data_root, name))
        dataset[name] = data_raw.copy().to(device)
        # loader[name] = DataLoader(dataset[name])

    # 모델 및 옵티마이저
    model = LinearGNN(mu=0.3, state_dimension=3).to(device) # 테스트버전이므로 임의로 설정했습니다.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 학습 및 검증
    for epoch in tqdm(range(1, 1001)):
        model.train()
        total_loss = 0
        for data in dataset["train.pt"]:
            data: Data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss: torch.Tensor = criterion(out, data.y.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 20 == 0:
            val_acc = evaluate(model, dataset["validate.pt"])
            print(f"[Epoch {epoch:4d}] Train Loss: {total_loss:.4f}  Val Acc: {val_acc:.4f}")

except Exception as e:
    logging.error("Exception occurred:\n%s", traceback.format_exc())
    sys.exit(1)