# test.py

import argparse
import time
import torch
import pandas as pd
from model import GAT, SiameseNet, ResourceConceptPredictNet, ResourceDependencyPredictNet
from utils import set_seed, evaluate
from load_data import load_data

parser = argparse.ArgumentParser()
parser.add_argument('--in_channels', type=int, default=300, help='Input channel size for GAT')
parser.add_argument('--out_channels1', type=int, default=128, help='Output channel size for first GAT layer')
parser.add_argument('--out_channels2', type=int, default=512, help='Output channel size for second GAT layer')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size for testing')
parser.add_argument('--seed', type=int, default=25, help='Random seed')
args = parser.parse_args()


def test(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    data, homogeneous_data = load_data(device)

    r_contain_c_test = pd.read_csv('data/train_test_data/MOOC/r_contain_c_data_test.csv', header=0)
    r_depend_on_r_test = pd.read_csv('data/train_test_data/MOOC/depend_on_r_data_test.csv', header=0)
    test_data_df = pd.read_csv("data/train_test_data/MOOC/depend_on_c_data_test.csv", header=0)
    test_data = [tuple(x) for x in test_data_df.to_numpy()]
    concept_df = pd.read_csv('data/preprocess_data/MOOC/concepts_index.csv', header=None)
    num_concepts = len(concept_df)

    gat = GAT(in_channels=args.in_channels, out_channels1=args.out_channels1, out_channels2=args.out_channels2).to(
        device)
    siamesenet = SiameseNet(input_dim=args.out_channels2).to(device)
    resource_concept_predictNet = ResourceDependencyPredictNet(input_dim=args.out_channels2).to(device)
    resource_dependency_predictNet = ResourceDependencyPredictNet(input_dim=args.out_channels2).to(device)

    model_name = f"GKROM_best_net.pth"
    checkpoint = torch.load(model_name)
    gat.load_state_dict(checkpoint['gat_state_dict'])
    siamesenet.load_state_dict(checkpoint['siamesenet_state_dict'])
    resource_concept_predictNet.load_state_dict(checkpoint['resource_concept_predictNet_state_dict'])
    resource_dependency_predictNet.load_state_dict(checkpoint['resource_dependency_predictNet_state_dict'])

    print("Testing!!!!")

    test_metrics = evaluate(args, gat, siamesenet, resource_concept_predictNet, resource_dependency_predictNet,
                            homogeneous_data, test_data, r_contain_c_test, r_depend_on_r_test, num_concepts, device)

    print(f'Test metrics (Concept Prerequisite Relation): {test_metrics["main_task"]}')
    print(f'Test metrics (Document-Concept Containment Relation): {test_metrics["resource_concept"]}')
    print(f'Test metrics (Document-Document Dependency Relation): {test_metrics["resource_dependency"]}')


if __name__ == '__main__':
    start_time = time.time()
    test(args)
    end_time = time.time()
    total_time = end_time - start_time
    print(f'Test time: {total_time:.2f} seconds\n')
