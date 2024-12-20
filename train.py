## train.py

import argparse
import torch
from torch import nn
import numpy as np
import time
import pandas as pd
from sklearn.utils import shuffle
from model import GAT, SiameseNet, ResourceConceptPredictNet, ResourceDependencyPredictNet
from utils import set_seed, evaluate
from load_data import load_data

parser = argparse.ArgumentParser()
parser.add_argument('--in_channels', type=int, default=300, help='Input channel size for GAT')
parser.add_argument('--out_channels1', type=int, default=128, help='Output channel size for first GAT layer')
parser.add_argument('--out_channels2', type=int, default=512, help='Output channel size for second GAT layer')
parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
parser.add_argument('--alpha', type=float, default=0.5, help='Alpha value for loss weighting')
parser.add_argument('--beta', type=float, default=0.5, help='Beta value for loss weighting')
parser.add_argument('--lr', type=float, default=0.000001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
parser.add_argument('--seed', type=int, default=25, help='Random seed')
args = parser.parse_args()


def train(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Load data
    data, homogeneous_data = load_data(device)
    r_contain_c_train = pd.read_csv('data/train_test_data/MOOC/r_contain_c_data_train.csv', header=0)
    r_contain_c_val = pd.read_csv('data/train_test_data/MOOC/r_contain_c_data_val.csv', header=0)
    r_depend_on_r_train = pd.read_csv('data/train_test_data/MOOC/depend_on_r_data_train.csv', header=0)
    r_depend_on_r_val = pd.read_csv('data/train_test_data/MOOC/depend_on_r_data_val.csv', header=0)
    train_data_df = pd.read_csv("data/train_test_data/MOOC/depend_on_c_data_train.csv", header=0)
    val_data_df = pd.read_csv("data/train_test_data/MOOC/depend_on_c_data_val.csv", header=0)
    train_data = [tuple(x) for x in train_data_df.to_numpy()]
    val_data = [tuple(x) for x in val_data_df.to_numpy()]
    concept_df = pd.read_csv('data/preprocess_data/MOOC/concepts_index.csv', header=None)
    num_concepts = len(concept_df)

    # Initialize models
    gat = GAT(in_channels=args.in_channels, out_channels1=args.out_channels1, out_channels2=args.out_channels2).to(
        device)
    siamesenet = SiameseNet(input_dim=args.out_channels2).to(device)
    resource_concept_predictNet = ResourceConceptPredictNet(input_dim=args.out_channels2).to(device)
    resource_dependency_predictNet = ResourceDependencyPredictNet(input_dim=args.out_channels2).to(device)

    # Set up optimizer and loss function
    optimizer = torch.optim.Adam(
        list(gat.parameters())
        + list(siamesenet.parameters())
        + list(resource_concept_predictNet.parameters())
        + list(resource_dependency_predictNet.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay)
    criterion = nn.BCELoss()

    print("Training!!!!")
    best_val_auc = 0

    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        X_train = np.array(shuffle(train_data, random_state=args.seed))
        total_loss = 0
        gat.train()
        siamesenet.train()
        resource_concept_predictNet.train()
        resource_dependency_predictNet.train()

        batch_idx = 0
        for i in range(X_train.shape[0] // args.batch_size):
            x = X_train[batch_idx * args.batch_size: batch_idx * args.batch_size + args.batch_size]
            batch_idx += 1
            c1, c2 = x[:, 0], x[:, 1]
            target = x[:, -1]

            optimizer.zero_grad()
            gat_output = gat(homogeneous_data.x.float(), homogeneous_data.edge_index)

            # Calculate main loss
            target = torch.tensor(target).to(device)
            preq_pred_prob = siamesenet(gat_output[c1], gat_output[c2])
            loss1 = criterion(preq_pred_prob, target[:, None].float())

            # Calculate auxiliary loss related to resource-concept containment
            r_contain_c_data = r_contain_c_train[r_contain_c_train['concept'].isin(c1.tolist() + c2.tolist())]
            if len(r_contain_c_data) > args.batch_size:
                r_contain_c_data = r_contain_c_data.sample(n=args.batch_size, random_state=args.seed)

            resources_for_aux1 = []

            if not r_contain_c_data.empty:
                resource, concept, result = zip(*r_contain_c_data.itertuples(index=False, name=None))
                resources_for_aux1.extend(resource)

                adjusted_resource_indices = [r + num_concepts for r in resource]
                concept_indices = [c for c in concept]
                result = [s for s in result]
                result = torch.tensor(result).to(device)
                resource_concept_output = resource_concept_predictNet(gat_output[adjusted_resource_indices],
                                                                      gat_output[concept_indices])
                loss2 = criterion(resource_concept_output, result[:, None].float())
            else:
                loss2 = torch.tensor(0.0, requires_grad=True).to(device)

            # Calculate auxiliary loss related to resource-resource dependency
            if resources_for_aux1:
                r_depend_on_r_data = r_depend_on_r_train[
                    r_depend_on_r_train['resource_1'].isin(resources_for_aux1)
                    | r_depend_on_r_train['resource_2'].isin(resources_for_aux1)]
                if len(r_depend_on_r_data) > args.batch_size:
                    r_depend_on_r_data = r_depend_on_r_data.sample(n=args.batch_size, random_state=args.seed)

                if not r_depend_on_r_data.empty:
                    resource1, resource2, result = zip(*r_depend_on_r_data.itertuples(index=False, name=None))
                    adjusted_resource1_indices = [r + num_concepts for r in resource1]
                    adjusted_resource2_indices = [r + num_concepts for r in resource2]
                    result = [s for s in result]
                    result = torch.tensor(result).to(device)
                    resource_dependency_output = resource_dependency_predictNet(gat_output[adjusted_resource1_indices],
                                                                                gat_output[adjusted_resource2_indices])
                    loss3 = criterion(resource_dependency_output, result[:, None].float())
                else:
                    loss3 = torch.tensor(0.0, requires_grad=True).to(device)
            else:
                loss3 = torch.tensor(0.0, requires_grad=True).to(device)

            # Total loss
            loss = loss1 + args.alpha * loss2 + args.beta * loss3
            total_loss += loss
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(gat.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(siamesenet.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(resource_concept_predictNet.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(resource_dependency_predictNet.parameters(), max_norm=1.0)

            optimizer.step()

        average_loss = (total_loss / X_train.shape[0]).float()
        print("epoch: {}, average loss: {}".format(epoch, average_loss.item()))

        # Evaluate on the training data
        train_metrics = evaluate(args, gat, siamesenet, resource_concept_predictNet, resource_dependency_predictNet,
                                 homogeneous_data, train_data, r_contain_c_train, r_depend_on_r_train, num_concepts,
                                 device)
        print(f'Train metrics (Concept Prerequisite Relation): {train_metrics["main_task"]}')
        print(f'Train metrics (Document-Concept Containment Relation): {train_metrics["resource_concept"]}')
        print(f'Train metrics (Document-Document Dependency Relation): {train_metrics["resource_dependency"]}')

        # Evaluate on the validation data
        val_metrics = evaluate(args, gat, siamesenet, resource_concept_predictNet, resource_dependency_predictNet,
                               homogeneous_data, val_data, r_contain_c_val, r_depend_on_r_val, num_concepts, device)
        print(f'Validation metrics (Concept Prerequisite Relation): {val_metrics["main_task"]}')
        print(f'Validation metrics (Document-Concept Containment Relation): {val_metrics["resource_concept"]}')
        print(f'Validation metrics (Document-Document Dependency Relation): {val_metrics["resource_dependency"]}')

        # Save model
        if val_metrics['main_task']['auc'] > best_val_auc:
            best_val_auc = val_metrics['main_task']['auc']
            model_name = f"GKROM_best_net.pth"
            torch.save({
                'gat_state_dict': gat.state_dict(),
                'siamesenet_state_dict': siamesenet.state_dict(),
                'resource_concept_predictNet_state_dict': resource_concept_predictNet.state_dict(),
                'resource_dependency_predictNet_state_dict': resource_dependency_predictNet.state_dict(),
            }, model_name)
            print(f'Saved new best model with val_auc: {best_val_auc:.4f}')
            print(f"Model parameters saved to {model_name}！")

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f'Epoch {epoch} duration: {epoch_duration:.2f} seconds\n')


if __name__ == '__main__':
    start_time = time.time()
    train(args)
    end_time = time.time()
    total_time = end_time - start_time
    print(f'Train time: {total_time:.2f} seconds\n')
    # train_time: 20504.34s
