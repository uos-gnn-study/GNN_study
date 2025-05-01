from pyscripts.data import SubgraphMatchingDataset
import argparse
    
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir")
args = parser.parse_args()
root: str = args.output_dir
dataset = SubgraphMatchingDataset(root=root)