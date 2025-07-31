import numpy as np
import os
import torch
import json
from torch_geometric.data import Dataset
from torch_geometric.datasets import TUDataset
from ast_to_pyg import ASTDataset

# class SimpleASTDataset(Dataset):
#     def __init__(self, data_list, transform=None, pre_transform=None):
#         super().__init__(None, transform, pre_transform)
#         self.data_list = data_list

#     def len(self):
#         return len(self.data_list)

#     def get(self, idx):
#         return self.data_list[idx]

# This function will now be simpler: it ensures files exist, loads vocabs, 
# and then the main logic relies on ASTDataset class itself.
def _load_vocabs_and_get_paths(root_dir, max_nodes_per_graph, n_patches_for_partition):
    processed_dir = os.path.join(root_dir, 'processed')
    
    base_name = 'ast_pyg_data'
    if max_nodes_per_graph:
        base_name += f'_max_nodes_{max_nodes_per_graph}'
    if n_patches_for_partition is not None and n_patches_for_partition > 0:
        # We need to check if GraphPartitionTransform was available during ast_to_pyg.py generation.
        # This filename logic should exactly match ASTDataset.processed_file_names property.
        # A more robust way would be for ASTDataset to have a static method to get expected filename.
        # For now, we replicate the logic, assuming GraphPartitionTransform was available if n_patches > 0.
        base_name += f'_n_patches_{n_patches_for_partition}'
    pt_filename = base_name + '.pt'
        
    data_path = os.path.join(processed_dir, pt_filename)
    vocab_path = os.path.join(processed_dir, 'vocabs.json')
    
    if not os.path.exists(data_path):
        # Construct the expected raw directories for a more helpful error message
        # These are defaults from ast_to_pyg.py if not passed to ASTDataset directly.
        # This part of the error message might need adjustment if raw_dirs are configurable.
        bad_ast_dir_example = os.path.join(os.path.dirname(root_dir), 'deobfuscate_ast_output_bad') # Guessing relative path
        good_ast_dir_example = os.path.join(os.path.dirname(root_dir), 'deobfuscate_ast_output_good')
        raise FileNotFoundError(f"Processed data file not found: {data_path}. "
                              f"Please run ast_to_pyg.py first. It expects raw data in directories like "
                              f"{bad_ast_dir_example} and {good_ast_dir_example} relative to your project root, "
                              f"and processes them into {processed_dir}.")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabs file not found: {vocab_path}. Please run ast_to_pyg.py first.")

    with open(vocab_path, 'r') as f:
        vocabs = json.load(f)
        
    node_type_vocab_size = vocabs['node_type_counter']
    node_name_vocab_size = vocabs['node_name_counter']
    node_value_vocab_size = vocabs['node_value_counter']
    
    # We don't load data_obj or slices here anymore. ASTDataset will do it.
    return data_path, vocab_path, node_type_vocab_size, node_name_vocab_size, node_value_vocab_size

def load_data(args):
    if args.data == 'AST_MALWARE': 
        if not hasattr(args, 'dataset_root_dir') or args.dataset_root_dir is None:
            args.dataset_root_dir = './data/my_ast_semantic_dataset' 
            print(f"Warning: --dataset_root_dir not specified for AST_MALWARE, using default: {args.dataset_root_dir}")
        
        m_nodes = getattr(args, 'max_nodes_per_graph', 1000)
        if args.max_nodes_per_graph is None:
            args.max_nodes_per_graph = m_nodes
        
        n_patches = getattr(args, 'n_patches', 8) # Default to 8 if not specified, as per model/ast_to_pyg common use
        if not hasattr(args, 'n_patches') or args.n_patches is None: # Explicitly set if was None
            args.n_patches = n_patches
            print(f"Warning: --n_patches not specified for AST_MALWARE, using default: {n_patches} for data loading and model.")

        _, _, type_vocab, name_vocab, value_vocab = _load_vocabs_and_get_paths(
            args.dataset_root_dir, 
            args.max_nodes_per_graph,
            n_patches
        )
        args.node_type_vocab_size = type_vocab
        args.node_name_vocab_size = name_vocab
        args.node_value_vocab_size = value_vocab

        # Instantiate ASTDataset. It will load the .pt file itself.
        # The raw_dir paths are not strictly needed by ASTDataset if the .pt file exists,
        # but good to pass if available for consistency or if .process() was ever triggered from here.
        # For now, assuming .pt exists, these are less critical for loading.
        # Defaults from ast_to_pyg.py if needed for a hypothetical .process() call from here:
        # bad_ast_dir_default = os.path.join(os.path.dirname(args.dataset_root_dir), 'deobfuscate_ast_output_bad')
        # good_ast_dir_default = os.path.join(os.path.dirname(args.dataset_root_dir), 'deobfuscate_ast_output_good')

        dataset = ASTDataset(
            root=args.dataset_root_dir, 
            # bad_ast_dir and good_ast_dir are primarily for .process() method.
            # If .pt file exists, ASTDataset.__init__ loads it and doesn't need these immediately.
            # However, ASTDataset constructor requires them.
            # We should provide placeholder or actual paths if known.
            # Assuming standard relative paths from ast_to_pyg.py for placeholder:
            bad_ast_dir='./deobfuscate_ast_output_bad', 
            good_ast_dir='./deobfuscate_ast_output_good',
            max_nodes_per_graph=args.max_nodes_per_graph,
            n_patches_for_partition=n_patches,
            # graph_partition_args are not explicitly handled by main.py args, so use ASTDataset defaults
            graph_partition_args=None 
        )
        return dataset
    else:
        raise ValueError(f"Unsupported dataset type: {args.data}. This script is now configured primarily for AST_MALWARE.")


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        # For a single Data object or an InMemoryDataset/Dataset of single graphs.
        # If data is an ASTDataset (InMemoryDataset), len(data) is the number of graphs.
        if isinstance(data, Dataset): # Check if it's a PyG Dataset (includes InMemoryDataset)
            return len(data)
        elif hasattr(data, 'x') and data.x is not None: # Fallback for a single Data object
             return 1 
        else: 
            return 0

