import time
import argparse
import random
import time
import numpy as np
from tqdm import tqdm, trange
from functools import reduce
import csv
import json

import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from transformers.optimization import get_cosine_schedule_with_warmup

from sklearn.model_selection import KFold

from eval_with_roc import eval_with_roc, print_evaluation_summary, eval

from data import load_data, num_graphs

import os


# Parser - add_argument
parser = argparse.ArgumentParser(description='DeCoda')

# data setting
parser.add_argument('--data', default='IMDB-MULTI', type=str,
                    choices=['DD', 'PTC_MR', 'NCI1', 'PROTEINS', 'IMDB-BINARY',
                             'IMDB-MULTI', 'MUTAG', 'COLLAB', 'ENZYMES', 'AST_MALWARE'],
                    help='dataset type')
parser.add_argument('--dataset_root_dir', type=str, default=None, 
                    help='Root directory for custom datasets like AST_MALWARE (contains processed/)')
parser.add_argument('--max_nodes_per_graph', type=int, default=None, 
                    help='Maximum number of nodes per graph for AST_MALWARE dataset, to load correct .pt file')

# model setting
parser.add_argument("--model", type=str,
                    default='DeCoda', choices=['DeCoda'])
parser.add_argument("--model-string", type=str, default='DeCoda')

# ROC evaluation settings
parser.add_argument('--use_roc_eval', action='store_true', help='使用ROC评估并生成可视化结果')
parser.add_argument('--roc_save_dir', type=str, default='./results', help='ROC曲线和评估结果保存目录')

# Embedding settings for AST_MALWARE (relevant if model handles embeddings)
parser.add_argument('--embedding_dim_type', type=int, default=32, help='Embedding dimension for AST node types')
parser.add_argument('--embedding_dim_name', type=int, default=64, help='Embedding dimension for AST node names/identifiers')
parser.add_argument('--embedding_dim_value', type=int, default=32, help='Embedding dimension for AST node literal values')

# fixed setting
parser.add_argument('--seed', type=int, default=42, help='seed')
parser.add_argument("--grad-norm", type=float, default=1.0)
parser.add_argument("--lr-schedule", action='store_true')
parser.add_argument("--normalize", action='store_true')
parser.add_argument('--num-epochs', default=300,
                    type=int, help='train epochs number')
parser.add_argument('--patience', type=int, default=30,
                    help='patience for earlystopping')


# training setting
parser.add_argument('--num-hidden', type=int, default=32, help='hidden size')
parser.add_argument('--batch-size', default=64,
                    type=int, help='train batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight-decay', type=float,
                    default=0.00001, help='weight decay')
parser.add_argument("--dropout", type=float, default=0.)

# gpu setting
parser.add_argument("--gpu", type=int, default=0)

# input transformation setting
parser.add_argument("--use-gnn", action='store_true')
parser.add_argument("--conv", type=str, default='GIN', choices=['GCN', 'GIN'])
parser.add_argument("--num-convs", type=int, default=1)

# hyper-parameter for model arch
parser.add_argument("--online", action='store_true')
parser.add_argument("--layernorm", action='store_true')
parser.add_argument("--remain-k1", action='store_true')
parser.add_argument("--diffQ", action='store_true')
parser.add_argument("--residual", type=str, default='cat',
                    choices=['None', 'cat', 'sum'])
parser.add_argument("--kernel_method", type=str,
                    default='elu', choices=['relu', 'elu'])
parser.add_argument("--deepset-layers", type=int, default=2)
parser.add_argument("--pos-enc-rw-dim", type=int, default=8)
parser.add_argument("--pos-enc-lap-dim", type=int, default=0)
parser.add_argument("--n-patches", type=int, default=8)
parser.add_argument("--prop-w-norm-on-coarsened", action='store_true')
parser.add_argument("--pos-enc-patch-rw-dim", type=int, default=0)
parser.add_argument("--pos-enc-patch-num-diff", type=int, default=-1)
parser.add_argument("--attention-based-readout", action='store_true')
parser.add_argument("--convex-linear-combination", action='store_true')


args = parser.parse_args()
print(args)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

use_cuda = args.gpu >= 0 and torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(args.gpu)
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'

torch.set_num_threads(1)

# Load data using the updated load_data function from data.py
dataset = load_data(args)

# ... 接下来是创建 DataLoader 的代码 ...
# train_loader = DataLoader(train_dataset, ...)

# Determine num_features and num_classes based on the dataset type
if args.data == 'AST_MALWARE':
    # For AST_MALWARE, num_features will be the sum of embedding dimensions after processing in the model
    # The model will handle the raw [type, name, value] IDs and convert them to embeddings.
    # So, the raw input feature count is 3 (type_id, name_id, value_id).
    # We can pass the individual vocab sizes and embedding dims via args to the model.
    args.num_features = 3 # Raw input feature: type_id, name_id, value_id
    args.num_classes = 2  # Binary classification: good/bad
    # avg_num_nodes might be useful for some model initializations or logging
    num_nodes_list = []
    half_len = len(dataset) // 2
    for i in range(half_len):
        num_nodes_list.append(dataset[i].num_nodes)
    if len(dataset) > half_len: # Handle the rest if any, to keep avg calculation somewhat similar
        for i in range(half_len, len(dataset)):
            num_nodes_list.append(dataset[i].num_nodes) # Or just use a subset
    args.avg_num_nodes = np.ceil(np.mean(num_nodes_list))
    print(f'# AST_MALWARE: [RAW FEATURES OD NODES]-{args.num_features} [NUM_CLASSES] Oka-{args.num_classes} [AVG_NODES] Oka-{args.avg_num_nodes}')
    # Vocab sizes are already set in args by load_data in data.py
else:
    # Logic for TUDatasets
    args.num_features, args.num_classes, args.avg_num_nodes = dataset.num_features, dataset.num_classes, np.ceil(
        np.mean([data.num_nodes for data in dataset]))
    print('# %s: [FEATURES]-%d [NUM_CLASSES]-%d [AVG_NODES]-%d' %
          (args.data, args.num_features, args.num_classes, args.avg_num_nodes))

print(f"Dataset: {args.data}")
print(f"Input transfrom: {'GNN' if args.use_gnn else 'MLP'}")
print(f'Metis Online: {args.online}')
print(f"Model: {args.model}")
print(f"Device: {args.device}")

overall_results = {
    'best_val_loss': [],
    'best_val_acc': [],
    'best_test_loss': [],
    'best_test_acc': [],
    'durations': []
}

# Prepare for 10-fold cross-validation
num_folds = 10
if args.data == 'AST_MALWARE':
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=args.seed)
    # We need a list of all indices to split. Assuming dataset is a list-like object or PyG Dataset.
    all_indices = list(range(len(dataset)))
    fold_iterator = kf.split(all_indices) # This will yield train_idx, test_idx for each fold
else:
    # For existing datasets, use the file-based fold indices
    # The original code had a slightly complex way of pairing val_fold_number, 
    # we might simplify or adjust if direct 10-fold is intended.
    # Assuming the files represent a standard 10-fold split where test_idx-F is the F-th test fold.
    # And train_idx-F is the F-th training fold (potentially excluding a validation set).
    # The original logic seemed to use test_idx-(fold-2) as validation, which is a bit unusual.
    # For simplicity, let's assume a direct 10-fold train/test if using files, 
    # or one might need to create separate validation splits.
    
    # We will iterate from 1 to 10 for fold_number as in the original code
    # and load corresponding train/test files directly.
    # The user needs to ensure these files correctly partition the data for 10 folds.
    fold_iterator = range(1, num_folds + 1)


# train_fold_iter = tqdm(range(1, 11), desc='Training') # Original tqdm iterator
# val_fold_iter = [i for i in range(1, 11)] # Original val fold logic

# Iterate over folds
# We use enumerate for kf.split to get a fold_number like behavior for logging if needed.
for fold_idx, current_fold_data in enumerate(tqdm(fold_iterator, total=num_folds, desc='Training Folds')):
    
    if args.data == 'AST_MALWARE':
        train_idxes, test_idxes = current_fold_data
        # AST_MALWARE currently doesn't have a pre-defined validation set strategy from files.
        # We will split train_idxes further into train and validation (e.g., 90/10 split).
        # Or, for simplicity in this step, use a portion of test_idxes as val_idxes if allowed,
        # but it's better to split train_idxes to avoid data leakage from test set into validation.
        
        # Let's split train_idxes into actual train and validation
        # Convert to numpy for easier splitting if not already
        train_idxes_np = np.array(train_idxes)
        np.random.shuffle(train_idxes_np) # Shuffle before splitting
        val_split_idx = int(len(train_idxes_np) * 0.1) # e.g., 10% of train for validation
        if val_split_idx == 0 and len(train_idxes_np) > 1: # Ensure val set is not empty if train set is small but >1
            val_split_idx = 1

        val_idxes_np = train_idxes_np[:val_split_idx]
        train_idxes_np = train_idxes_np[val_split_idx:]
        
        train_idxes = torch.from_numpy(train_idxes_np).long()
        val_idxes = torch.from_numpy(val_idxes_np).long()
        test_idxes = torch.from_numpy(np.array(test_idxes)).long()
        fold_number_for_print = fold_idx + 1 # for logging compatibility

    else: # For TUDatasets using file-based splits
        fold_number = current_fold_data # Here current_fold_data is just the fold number (1 to 10)
        fold_number_for_print = fold_number
        
        # Original logic for loading TUDataset fold files:
        # It used fold_number for train/test and fold_number-2 for validation.
        # This can lead to issues for fold_number 1 or 2 for validation.
        # We should use a more standard way if possible, or clearly define how val_idx is obtained.
        # For now, we adapt the original logic but with a small fix for val_fold_number.
        val_fold_number = (fold_number - 3 + num_folds) % num_folds + 1 # e.g. fold 1 uses fold 9, fold 2 uses fold 10
                                                                    # This is a guess at intention; typical CV would resplit train.

        # Path construction for TUDatasets
        train_idx_path = f'./datasets/{args.data}/10fold_idx/train_idx-{fold_number}.txt'
        # Original val_idx_path used test_idx file from a different fold for validation
        val_idx_path = f'./datasets/{args.data}/10fold_idx/test_idx-{val_fold_number}.txt' 
        test_idx_path = f'./datasets/{args.data}/10fold_idx/test_idx-{fold_number}.txt'

        try:
            train_idxes = torch.as_tensor(np.loadtxt(train_idx_path, dtype=np.int32), dtype=torch.long)
            val_idxes = torch.as_tensor(np.loadtxt(val_idx_path, dtype=np.int32), dtype=torch.long)
            test_idxes = torch.as_tensor(np.loadtxt(test_idx_path, dtype=np.int32), dtype=torch.long)
        except FileNotFoundError as e:
            print(f"Error loading fold files for {args.data}: {e}")
            print("Please ensure 10-fold index files (train_idx-F.txt, test_idx-F.txt) exist in ./datasets/<DATASET_NAME>/10fold_idx/")
            continue # Skip this fold or raise error

        # Original logic to ensure train and val are disjoint if val comes from a train split originally.
        # If val_idxes are taken from a test split of another fold, this is different.
        # The original code did: train_idxes = torch.as_tensor(np.setdiff1d(train_idxes, val_idxes))
        # This implies val_idxes were expected to be a subset of the original train_idxes from its file.
        # Given val_idxes now come from a test_idx file, they should ideally be disjoint from the current fold's train/test.
        # We need to be careful about data leakage or re-using test data for validation.

        # For now, we will assume the loaded train_idxes, val_idxes, and test_idxes are as intended by original setup.
        # The most problematic part is val_idxes potentially overlapping with test_idxes of other folds if not careful.
        # A common good practice: load main train_idx, test_idx for the fold. Then split train_idx into train and val.

    # The original code to check all_idxes and setdiff for train_idxes vs val_idxes
    # might not be directly applicable or needs careful thought with KFold
    # For KFold, train and test are already disjoint by definition.
    # For file-based, it depends on how files were created.
    if args.data != 'AST_MALWARE': # Original check was more general
        all_idxes_check = reduce(np.union1d, (train_idxes.numpy(), val_idxes.numpy(), test_idxes.numpy()))
        if len(all_idxes_check) > len(dataset):
            print(f"Warning: Union of train/val/test indices ({len(all_idxes_check)}) is larger than dataset size ({len(dataset)}) for fold {fold_number_for_print}")
        # Original: train_idxes = torch.as_tensor(np.setdiff1d(train_idxes, val_idxes))
        # This line is problematic if val_idxes are from a different fold's test set.
        # It should be: split current fold's train_idxes into train and validation.
        # We will do this for AST_MALWARE. For TUDatasets, we keep original behavior for now, assuming files are set up for it.
        if not np.intersect1d(train_idxes.numpy(), val_idxes.numpy()).size == 0:
            print(f"Warning: Train and Val indices overlap for fold {fold_number_for_print}. This may be intended by original TUDataset setup if val is a subset of train file.")
            # If val_idxes from test_idx-fold_val.txt is used, ensure it's not from current train_idxes.
            # The safest is to split the current train_idxes into train/val if using file based method too.


    train_set, val_set, test_set = dataset[train_idxes], dataset[val_idxes], dataset[test_idxes]

    if not args.online:
        train_set = [x for x in train_set]
    val_set = [x for x in val_set]
    test_set = [x for x in test_set]

    train_loader = DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(
        dataset=val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(
        dataset=test_set, batch_size=args.batch_size, shuffle=False)

    if args.model == 'DeCoda':
        if args.convex_linear_combination:
            from DeCoda_N2C_L import DeCoda
        else:
            from DeCoda_N2C_T import DeCoda
        model = DeCoda(args)
    else:
        raise ValueError("Model Name <{}> is Unknown".format(args.model))

    if use_cuda:
        model.to(args.device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_schedule:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=args.num_epochs//10, num_training_steps=args.num_epochs)

    patience = 0
    best_loss = 1e9
    best_val_acc = 0
    best_test_loss = 1e9
    best_test_acc = 0

    t_start = time.perf_counter()

    epoch_iterator = trange(0, (args.num_epochs), desc='[Epoch]', position=1)
    for epoch in epoch_iterator:
        model.train()
        total_loss = 0
        for batch_idx, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(args.device)
            out = model(data)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            total_loss += loss.item() * num_graphs(data)
            optimizer.step()

            if args.lr_schedule:
                scheduler.step()

        total_loss = total_loss / len(train_loader.dataset)

        val_acc, val_loss = eval(val_loader, model, args)
        test_acc, test_loss = eval(test_loader, model, args)

        if val_loss < best_loss:
            best_loss = val_loss
            best_val_acc = val_acc
            best_val_epoch = epoch
            patience = 0

            best_test_acc = test_acc
            best_test_loss = test_loss
            
            # 使用 eval_with_roc 获取详细的评估结果
            if args.use_roc_eval:
                # 为数据集创建保存目录
                save_dir = os.path.join(args.roc_save_dir, args.data, f'fold_{fold_number_for_print}')
                class_names = ['Benign', 'Malicious'] if args.data == 'AST_MALWARE' else [str(i) for i in range(args.num_classes)]
                test_results = eval_with_roc(test_loader, model, args, 
                                           class_names=class_names, 
                                           save_plots=True, 
                                           save_dir=save_dir)
                # 打印详细的评估结果
                print_evaluation_summary(test_results, class_names=class_names)
                
                # 保存每个fold的TPR@FPR值到CSV文件
                if test_results.get('tpr_at_fpr_levels'):
                    import csv
                    os.makedirs(save_dir, exist_ok=True)
                    with open(os.path.join(save_dir, f'fold_{fold_number_for_print}_tpr_at_fpr_values.csv'), 'w', newline='') as csvfile:
                        fieldnames = ['fpr_level', 'tpr_value']
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        for key, value in test_results['tpr_at_fpr_levels'].items():
                            writer.writerow({'fpr_level': key, 'tpr_value': value})
                    
                    # 将TPR@FPR值添加到fold的评估结果JSON中
                    import json
                    with open(os.path.join(save_dir, f'fold_{fold_number_for_print}_metrics.json'), 'w') as f:
                        json.dump(test_results, f, indent=4)
        else:
            patience += 1

        epoch_iterator.set_description('[Val: Fold %d-Epoch %d] TrL: %.2f VaL: %.2f VaAcc: %.2f TestAcc: %.2f' % (
            fold_number_for_print, epoch, total_loss, val_loss, val_acc, test_acc))
        epoch_iterator.refresh()

        if patience > args.patience:
            break

    t_end = time.perf_counter()

    overall_results['durations'].append(t_end - t_start)
    overall_results['best_val_loss'].append(best_loss)
    overall_results['best_val_acc'].append(best_val_acc)
    overall_results['best_test_loss'].append(best_test_loss)
    overall_results['best_test_acc'].append(best_test_acc)

    print("[Test: Fold {}] Test Acc: {} with Time: {}".format(
        fold_number_for_print, best_test_acc, (t_end - t_start)))

print("Overall result - overall_best_val: {} with std: {}; overall_best_test: {} with std: {}\n".format(
    np.array(overall_results['best_val_acc']).mean(),
    np.array(overall_results['best_val_acc']).std(),
    np.array(overall_results['best_test_acc']).mean(),
    np.array(overall_results['best_test_acc']).std()
))

# 如果使用ROC评估，汇总所有fold的TPR@FPR值
if args.use_roc_eval:
    # 创建一个字典来存储所有fold的TPR@FPR值
    all_tpr_at_fpr = {f'tpr@fpr={fpr}': [] for fpr in [0.0001, 0.001, 0.01, 0.1]}
    
    # 遍历所有fold目录，读取TPR@FPR值
    for fold_idx in range(1, num_folds + 1):
        fold_dir = os.path.join(args.roc_save_dir, args.data, f'fold_{fold_idx}')
        metrics_file = os.path.join(fold_dir, f'fold_{fold_idx}_metrics.json')
        
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                fold_metrics = json.load(f)
                
            if fold_metrics.get('tpr_at_fpr_levels'):
                for key, value in fold_metrics['tpr_at_fpr_levels'].items():
                    if key in all_tpr_at_fpr:
                        all_tpr_at_fpr[key].append(value)
    
    # 计算平均值和标准差
    tpr_summary = {}
    for key, values in all_tpr_at_fpr.items():
        if values:
            tpr_summary[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
    
    # 打印TPR@FPR汇总表
    if tpr_summary:
        print("\n" + "="*60)
        print("📊 TPR@FPR SUMMARY ACROSS ALL FOLDS")
        print("="*60)
        print(f"{'FPR Level':<15} {'Mean TPR':<15} {'Std Dev':<15}")
        print("-"*45)
        
        for key, stats in tpr_summary.items():
            print(f"{key:<15} {stats['mean']:.6f}      {stats['std']:.6f}")
        
        # 保存TPR@FPR汇总表到CSV文件
        summary_dir = os.path.join(args.roc_save_dir, args.data)
        os.makedirs(summary_dir, exist_ok=True)
        
        with open(os.path.join(summary_dir, 'tpr_at_fpr_summary.csv'), 'w', newline='') as csvfile:
            fieldnames = ['fpr_level', 'mean_tpr', 'std_tpr'] + [f'fold_{i}' for i in range(1, num_folds + 1)]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for key, stats in tpr_summary.items():
                row = {
                    'fpr_level': key,
                    'mean_tpr': stats['mean'],
                    'std_tpr': stats['std']
                }
                
                # 添加每个fold的值
                for i, value in enumerate(stats['values'], 1):
                    if i <= num_folds:
                        row[f'fold_{i}'] = value
                
                writer.writerow(row)
        
        print(f"\nTPR@FPR汇总表已保存到 {os.path.join(summary_dir, 'tpr_at_fpr_summary.csv')}")

# 如果使用ROC评估，对最后一个fold的模型进行最终评估并生成ROC曲线
if args.use_roc_eval:
    # 加载最后一个fold的模型（或者可以保存每个fold的最佳模型，这里简化处理）
    print("\n生成最终的ROC评估结果...")
    
    # 创建保存目录
    final_save_dir = os.path.join(args.roc_save_dir, args.data, 'final_results')
    os.makedirs(final_save_dir, exist_ok=True)
    
    # 确定类名
    class_names = ['Benign', 'Malicious'] if args.data == 'AST_MALWARE' else [str(i) for i in range(args.num_classes)]
    
    # 对测试集进行评估
    final_test_results = eval_with_roc(test_loader, model, args,
                                     class_names=class_names,
                                     save_plots=True,
                                     save_dir=final_save_dir)
    
    # 打印最终评估结果
    print("\n最终测试集评估结果:")
    print_evaluation_summary(final_test_results, class_names=class_names)
    
    # 保存评估指标到文件
    import json
    
    # 提取特定FPR水平下的TPR值，并单独保存
    if final_test_results.get('tpr_at_fpr_levels'):
        tpr_values = {}
        for fpr_level in [0.0001, 0.001, 0.01, 0.1]:
            key = f'tpr@fpr={fpr_level}'
            if key in final_test_results['tpr_at_fpr_levels']:
                tpr_values[key] = final_test_results['tpr_at_fpr_levels'][key]
        
        # 将TPR@FPR值添加到最终结果中
        final_test_results['tpr_values'] = tpr_values
        
        # 单独保存TPR@FPR值到CSV文件，方便后续分析
        import csv
        with open(os.path.join(final_save_dir, 'tpr_at_fpr_values.csv'), 'w', newline='') as csvfile:
            fieldnames = ['fpr_level', 'tpr_value']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for fpr_level, tpr_value in tpr_values.items():
                writer.writerow({'fpr_level': fpr_level, 'tpr_value': tpr_value})
    
    with open(os.path.join(final_save_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(final_test_results, f, indent=4)
    
    print(f"\n评估结果已保存到 {final_save_dir}")

