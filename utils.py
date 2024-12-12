# utils.py

import numpy as np
import random
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, \
    average_precision_score


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_true_label(ccf, p_i):
    p_i_list = []
    ccf_list = []
    labels = []

    for each_str in p_i:
        id_pair = each_str[:-1].split(" ")
        p_i_list.append(tuple(map(int, id_pair)))

    for i in ccf.T:
        ccf_list.append(tuple(i))

    for each_pair in ccf_list:
        if each_pair in p_i_list:
            labels.append(1)
        else:
            labels.append(0)

    return labels


def evaluate_main_task(args, siamesenet, gat_output, data, device):
    batch_idx = 0
    X_data = np.array(data)
    batch_pred_prob = []
    batch_target_label = []

    for i in range(X_data.shape[0] // args.batch_size):
        x = X_data[batch_idx * args.batch_size: batch_idx * args.batch_size + args.batch_size]
        batch_idx += 1

        c1, c2 = x[:, 0], x[:, 1]
        target = x[:, -1]
        target = torch.tensor(target).to(device)
        batch_target_label.append(target.data.cpu().tolist())

        preq_pred_prob = siamesenet(gat_output[c1], gat_output[c2])
        batch_pred_prob.append(preq_pred_prob[:, 0].data.cpu().tolist())

    target_label = [each_target for each_list in batch_target_label for each_target in each_list]
    pred_prob = [each_prob for each_list in batch_pred_prob for each_prob in each_list]
    pred_label = [1 if prob >= 0.5 else 0 for prob in pred_prob]

    ACC = accuracy_score(target_label, pred_label)
    F1 = f1_score(target_label, pred_label)
    pre = precision_score(target_label, pred_label, zero_division=1)
    re = recall_score(target_label, pred_label)
    auc = roc_auc_score(target_label, pred_prob)
    ap_score = average_precision_score(target_label, pred_prob)

    return {
        'ACC': ACC,
        'F1': F1,
        'AUC': auc
    }


def evaluate_resource_concept_task(args, resource_concept_predictNet, gat_output, r_contain_c, num_concepts, device):
    batch_idx = 0
    X_data = np.array(r_contain_c)
    batch_pred_prob = []
    batch_target_label = []

    for i in range(X_data.shape[0] // args.batch_size):
        x = X_data[batch_idx * args.batch_size: batch_idx * args.batch_size + args.batch_size]
        batch_idx += 1

        resource, concept, result = x[:, 0], x[:, 1], x[:, 2]
        adjusted_resource_indices = [r + num_concepts for r in resource]
        result = torch.tensor(result).to(device)
        batch_target_label.append(result.data.cpu().tolist())

        rc_pred_prob = resource_concept_predictNet(gat_output[adjusted_resource_indices], gat_output[concept])
        batch_pred_prob.append(rc_pred_prob[:, 0].data.cpu().tolist())

    target_label = [each_target for each_list in batch_target_label for each_target in each_list]
    pred_prob = [each_prob for each_list in batch_pred_prob for each_prob in each_list]
    pred_label = [1 if prob >= 0.5 else 0 for prob in pred_prob]

    ACC = accuracy_score(target_label, pred_label)
    F1 = f1_score(target_label, pred_label)
    pre = precision_score(target_label, pred_label, zero_division=1)
    re = recall_score(target_label, pred_label)
    auc = roc_auc_score(target_label, pred_prob)
    ap_score = average_precision_score(target_label, pred_prob)

    return {
        'ACC': ACC,
        'F1': F1,
        'AUC': auc
    }


def evaluate_resource_dependency_task(args, resource_dependency_predictNet, gat_output, r_depend_on_r, num_concepts,
                                      device):
    batch_idx = 0
    X_data = np.array(r_depend_on_r)
    batch_pred_prob = []
    batch_target_label = []

    for i in range(X_data.shape[0] // args.batch_size):
        x = X_data[batch_idx * args.batch_size: batch_idx * args.batch_size + args.batch_size]
        batch_idx += 1

        resource1, resource2, result = x[:, 0], x[:, 1], x[:, 2]
        adjusted_resource1_indices = [r + num_concepts for r in resource1]
        adjusted_resource2_indices = [r + num_concepts for r in resource2]
        result = torch.tensor(result).to(device)
        batch_target_label.append(result.data.cpu().tolist())

        rr_pred_prob = resource_dependency_predictNet(gat_output[adjusted_resource1_indices],
                                                      gat_output[adjusted_resource2_indices])
        batch_pred_prob.append(rr_pred_prob[:, 0].data.cpu().tolist())

    target_label = [each_target for each_list in batch_target_label for each_target in each_list]
    pred_prob = [each_prob for each_list in batch_pred_prob for each_prob in each_list]
    pred_label = [1 if prob >= 0.5 else 0 for prob in pred_prob]

    ACC = accuracy_score(target_label, pred_label)
    F1 = f1_score(target_label, pred_label)
    pre = precision_score(target_label, pred_label, zero_division=1)
    re = recall_score(target_label, pred_label)
    auc = roc_auc_score(target_label, pred_prob)
    ap_score = average_precision_score(target_label, pred_prob)

    return {
        'ACC': ACC,
        'F1': F1,
        'AUC': auc
    }


def evaluate(args, gat, siamesenet, resource_concept_predictNet, resource_dependency_predictNet, homogeneous_data, data,
             r_contain_c, r_depend_on_r, num_concepts, device):
    gat_output = gat(homogeneous_data.x.float(), homogeneous_data.edge_index)

    # Evaluate the main task
    main_task_metrics = evaluate_main_task(args, siamesenet, gat_output, data, device)

    # Evaluate the first auxiliary task (Resource-Concept)
    rc_metrics = evaluate_resource_concept_task(args, resource_concept_predictNet, gat_output, r_contain_c,
                                                num_concepts, device)

    # Evaluate the second auxiliary task (Resource-Resource)
    rr_metrics = evaluate_resource_dependency_task(args, resource_dependency_predictNet, gat_output, r_depend_on_r,
                                                   num_concepts, device)

    return {
        'main_task': main_task_metrics,
        'resource_concept': rc_metrics,
        'resource_dependency': rr_metrics,
    }
