import os
import json
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

# --- DEBUGGING IMPORTS ---
print("[ast_to_pyg DEBUG] Initializing GraphPartitionTransform and SubgraphsData to None by default...")
GraphPartitionTransform = None
SubgraphsData = None

print("[ast_to_pyg DEBUG] Attempting to import GraphPartitionTransform from dataset_transform...")
try:
    from dataset_transform import GraphPartitionTransform as GPT_imported
    GraphPartitionTransform = GPT_imported # Assign if import successful
    print(f"[ast_to_pyg DEBUG] Successfully ran import line for GraphPartitionTransform. Object is: {GraphPartitionTransform}")
    if GraphPartitionTransform is None:
        print("[ast_to_pyg DEBUG] WARNING: GraphPartitionTransform is None immediately after import line!")
except ImportError as e_gpt:
    print(f"[ast_to_pyg DEBUG] ImportError during GraphPartitionTransform import: {e_gpt}")
except Exception as e_gpt_other:
    print(f"[ast_to_pyg DEBUG] Other Exception during GraphPartitionTransform import: {e_gpt_other}")

print("[ast_to_pyg DEBUG] Attempting to import SubgraphsData from graph_partition...")
try:
    from graph_partition import SubgraphsData as SD_imported
    SubgraphsData = SD_imported # Assign if import successful
    print(f"[ast_to_pyg DEBUG] Successfully ran import line for SubgraphsData. Object is: {SubgraphsData}")
    if SubgraphsData is None:
        print("[ast_to_pyg DEBUG] WARNING: SubgraphsData is None immediately after import line!")
except ImportError as e_sd:
    print(f"[ast_to_pyg DEBUG] ImportError during SubgraphsData import: {e_sd}")
except Exception as e_sd_other:
    print(f"[ast_to_pyg DEBUG] Other Exception during SubgraphsData import: {e_sd_other}")
# --- END DEBUGGING IMPORTS ---

# --- 词汇表和计数器 ---
# 使用字典来存储词汇表，这样可以在 ASTDataset 的 init 中重新初始化
vocab_data = {
    'node_type_vocab': {},
    'node_type_counter': 0,
    'node_name_vocab': {}, # 用于Identifier的name
    'node_name_counter': 0,
    'node_value_vocab': {},# 用于Literal的value (或其类型)
    'node_value_counter': 0
}

# 特殊标记的ID
PAD_ID = 0 # 用于填充名称和值（如果不存在）
UNKNOWN_ID = 1 # 用于词汇表中未见过的名称/值 (如果选择不动态添加的话)

def _initialize_vocabs():
    global vocab_data
    vocab_data['node_type_vocab'] = {}
    vocab_data['node_type_counter'] = 0
    vocab_data['node_name_vocab'] = {'<PAD>': PAD_ID, '<UNK>': UNKNOWN_ID}
    vocab_data['node_name_counter'] = 2
    vocab_data['node_value_vocab'] = {'<PAD>': PAD_ID, '<UNK>': UNKNOWN_ID}
    vocab_data['node_value_counter'] = 2


def get_node_type_id(node_type):
    global vocab_data
    if node_type not in vocab_data['node_type_vocab']:
        vocab_data['node_type_vocab'][node_type] = vocab_data['node_type_counter']
        vocab_data['node_type_counter'] += 1
    return vocab_data['node_type_vocab'][node_type]

def get_node_name_id(name):
    global vocab_data
    if name not in vocab_data['node_name_vocab']:
        vocab_data['node_name_vocab'][name] = vocab_data['node_name_counter']
        vocab_data['node_name_counter'] += 1
    return vocab_data['node_name_vocab'][name]

def get_node_value_id(value):
    global vocab_data
    # 对于Literal，我们可以使用其值的类型或者值本身（如果值的种类可控）
    # 这里简化为使用值的字符串表示的类型，对于复杂对象可能需要更细致处理
    value_repr = type(value).__name__ if not isinstance(value, (str, int, float, bool)) else str(value)
    
    # 限制value词汇表的大小，避免过多稀有值
    if len(vocab_data['node_value_vocab']) > 5000 and value_repr not in vocab_data['node_value_vocab']: # 示例上限
        return vocab_data['node_value_vocab']['<UNK>']

    if value_repr not in vocab_data['node_value_vocab']:
        vocab_data['node_value_vocab'][value_repr] = vocab_data['node_value_counter']
        vocab_data['node_value_counter'] += 1
    return vocab_data['node_value_vocab'][value_repr]

# 边类型定义 (示例)
EDGE_TYPE_AST = 0
# EDGE_TYPE_CFG = 1 # 占位符 (不再需要)
# EDGE_TYPE_DFG = 2 # 占位符 (不再需要)


def ast_json_to_graph_data(json_data, label, max_nodes=None):
    node_features = []  # 存储每个图节点的特征 [type_id, name_id, value_id]
    edges_src = []      # 存储边的源节点ID
    edges_dst = []      # 存储边的目标节点ID
    edge_attributes = []# 存储边的属性 (例如边的类型)
    
    node_id_counter = 0
    # node_map = {} # id(ast_node_json) 不可靠，因为json.load会创建新对象

    # CFG/DFG 边 (占位符 - 实际构建需要复杂逻辑)
    # cfg_edges = [] # list of (src_node_id, dst_node_id)
    # dfg_edges = [] # list of (src_node_id, dst_node_id)

    def traverse_ast(ast_node_json, parent_node_id=None):
        nonlocal node_id_counter
        
        if max_nodes is not None and node_id_counter >= max_nodes:
            return None

        if not isinstance(ast_node_json, dict) or 'type' not in ast_node_json:
            if isinstance(ast_node_json, list):
                for item in ast_node_json:
                    if max_nodes is not None and node_id_counter >= max_nodes:
                        break 
                    traverse_ast(item, parent_node_id)
            return None

        current_node_id = node_id_counter
        node_id_counter += 1
        
        node_type = ast_node_json.get('type', 'UnknownType')
        type_id = get_node_type_id(node_type)
        
        name_id = PAD_ID
        if node_type == 'Identifier' and 'name' in ast_node_json:
            name_id = get_node_name_id(ast_node_json['name'])
        
        value_id = PAD_ID
        if node_type == 'Literal' and 'value' in ast_node_json:
            value_id = get_node_value_id(ast_node_json['value'])
            
        node_features.append([type_id, name_id, value_id])
        
        if parent_node_id is not None:
            edges_src.append(parent_node_id)
            edges_dst.append(current_node_id)
            edge_attributes.append([EDGE_TYPE_AST]) # AST 边
            
        # 递归遍历子节点
        for key, value in ast_node_json.items():
            # 跳过非子节点的元数据或已提取为特征的字段
            if key in ['type', 'range', 'loc', 'raw', 'name', 'value', 'leadingComments', 'trailingComments', 'comments']:
                continue
            
            if max_nodes is not None and node_id_counter >= max_nodes:
                break

            if isinstance(value, dict):
                traverse_ast(value, current_node_id)
            elif isinstance(value, list):
                for child_node in value:
                    if max_nodes is not None and node_id_counter >= max_nodes:
                        break
                    if isinstance(child_node, dict):
                         traverse_ast(child_node, current_node_id)
        return current_node_id # 返回当前节点的ID，可能用于CFG/DFG构建

    # 从根节点开始遍历
    root_node_id = traverse_ast(json_data)


    if not node_features:
        return Data(x=torch.empty((0, 3), dtype=torch.long), # 3 特征: type, name, value
                    edge_index=torch.empty((2, 0), dtype=torch.long),
                    edge_attr=torch.empty((0,1), dtype=torch.long),
                    y=torch.tensor([label], dtype=torch.long))

    x = torch.tensor(node_features, dtype=torch.long)
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_attributes, dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([label], dtype=torch.long))
    data.num_nodes = node_id_counter #显式设置节点数，尤其在max_nodes截断时
    return data


class ASTDataset(InMemoryDataset):
    def __init__(self, root, bad_ast_dir, good_ast_dir, max_nodes_per_graph=None, 
                 n_patches_for_partition=None, # New parameter for n_patches
                 graph_partition_args=None, # For other GraphPartitionTransform args
                 transform=None, pre_transform=None):
        self.bad_ast_dir = bad_ast_dir
        self.good_ast_dir = good_ast_dir
        self.max_nodes_per_graph = max_nodes_per_graph
        self.n_patches_for_partition = n_patches_for_partition
        self.graph_partition_args = graph_partition_args if graph_partition_args is not None else {}

        _initialize_vocabs() 
        
        # Initialize pre_transform list
        actual_pre_transforms = []
        if pre_transform is not None:
            if isinstance(pre_transform, list):
                actual_pre_transforms.extend(pre_transform)
            else:
                actual_pre_transforms.append(pre_transform)

        # Add GraphPartitionTransform if n_patches is specified
        print(f"[ASTDataset.__init__ DEBUG] n_patches_for_partition = {self.n_patches_for_partition}") # 调试 n_patches 值
        if self.n_patches_for_partition is not None and self.n_patches_for_partition > 0:
            print(f"[ASTDataset.__init__ DEBUG] Condition for n_patches met (value: {self.n_patches_for_partition}).")
            # 这里的 GraphPartitionTransform 引用的是 ast_to_pyg.py 文件顶部的全局（模块级）变量
            print(f"[ASTDataset.__init__ DEBUG] Checking GraphPartitionTransform object before use: {GraphPartitionTransform}") # 调试导入的对象
            
            if GraphPartitionTransform is None:
                print("[ASTDataset.__init__ CRITICAL WARNING] GraphPartitionTransform is None at the point of use! Cannot apply partitioning. "
                      "This means the import at the top of ast_to_pyg.py failed to assign the class correctly. "
                      "Check the [ast_to_pyg DEBUG] messages near the import statements.")
                # 如果 GraphPartitionTransform 为 None，则 GraphPartitionTransform(...) 会导致 TypeError，这里只是提前警告
            else:
                print(f"Attempting to apply GraphPartitionTransform with n_patches={self.n_patches_for_partition}")
                gpt_args = {
                    'metis': self.graph_partition_args.get('metis', True), 
                    'drop_rate': self.graph_partition_args.get('drop_rate', 0.0),
                    'num_hops': self.graph_partition_args.get('num_hops', 1),
                    'is_directed': self.graph_partition_args.get('is_directed', False),
                    'patch_rw_dim': self.graph_partition_args.get('patch_rw_dim', 0),
                    'patch_num_diff': self.graph_partition_args.get('patch_num_diff', 0)
                }
                try:
                    # 尝试实例化，如果 GraphPartitionTransform 是 None，这里会抛出 TypeError
                    partition_transform = GraphPartitionTransform(n_patches=self.n_patches_for_partition, **gpt_args)
                    actual_pre_transforms.append(partition_transform)
                    print(f"Successfully added GraphPartitionTransform to pre_transforms list.")
                except TypeError as e_type:
                    print(f"[ASTDataset.__init__ ERROR] TypeError during GraphPartitionTransform instantiation: {e_type}. "
                          "This usually means GraphPartitionTransform was None (import failed). Skipping partitioning.")
                except Exception as e_instantiate:
                    print(f"[ASTDataset.__init__ ERROR] Failed to instantiate or append GraphPartitionTransform: {e_instantiate}. "
                          "Skipping graph partitioning.")
        else:
            print(f"[ASTDataset.__init__ DEBUG] Skipping GraphPartitionTransform because n_patches_for_partition ({self.n_patches_for_partition}) is not a positive integer.")
        
        # Use a Compose if multiple pre_transforms, else just the single one or None
        if len(actual_pre_transforms) > 1:
            from torch_geometric.transforms import Compose
            final_pre_transform = Compose(actual_pre_transforms)
        elif len(actual_pre_transforms) == 1:
            final_pre_transform = actual_pre_transforms[0]
        else:
            final_pre_transform = None
            
        super().__init__(root, transform, pre_transform=final_pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        raw_files = []
        for dir_path in [self.bad_ast_dir, self.good_ast_dir]:
            if os.path.exists(dir_path):
                for subdir, _, files in os.walk(dir_path):
                    for file_name in files:
                        if file_name.endswith('.json'):
                             raw_files.append(os.path.join(subdir, file_name))
        if not raw_files:
            return ['placeholder.json'] 
        return raw_files

    @property
    def processed_file_names(self):
        base_name = 'ast_pyg_data'
        if self.max_nodes_per_graph:
            base_name += f'_max_nodes_{self.max_nodes_per_graph}'
        if self.n_patches_for_partition is not None and self.n_patches_for_partition > 0 and GraphPartitionTransform is not None:
            base_name += f'_n_patches_{self.n_patches_for_partition}'
            # Potentially add other gpt_args to filename if they vary, e.g., self.graph_partition_args.get('metis')
        return [base_name + '.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        
        print(f"Processing bad ASTs from: {self.bad_ast_dir}")
        if os.path.exists(self.bad_ast_dir):
            for subdir, _, files in os.walk(self.bad_ast_dir):
                # Sort files to ensure deterministic order for vocab creation if it matters
                # files.sort() 
                for file_name in tqdm(files, desc=f"Bad files in {os.path.basename(subdir)} ({len(files)} files)"):
                    if file_name.endswith('.json'):
                        file_path = os.path.join(subdir, file_name)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                ast_data_json = json.load(f)
                            data = ast_json_to_graph_data(ast_data_json, label=1, max_nodes=self.max_nodes_per_graph)
                            if data.num_nodes > 0 :
                                data_list.append(data)
                            else:
                                print(f"Skipping graph from file {file_path} due to zero nodes.")
                        except json.JSONDecodeError:
                            print(f"Warning: Could not decode JSON from {file_path}")
                        except Exception as e:
                            print(f"Warning: Error processing file {file_path}: {e}")
        else:
            print(f"Warning: Directory not found: {self.bad_ast_dir}")

        print(f"Processing good ASTs from: {self.good_ast_dir}")
        if os.path.exists(self.good_ast_dir):
            for subdir, _, files in os.walk(self.good_ast_dir):
                # files.sort()
                for file_name in tqdm(files, desc=f"Good files in {os.path.basename(subdir)} ({len(files)} files)"):
                    if file_name.endswith('.json'):
                        file_path = os.path.join(subdir, file_name)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                ast_data_json = json.load(f)
                            data = ast_json_to_graph_data(ast_data_json, label=0, max_nodes=self.max_nodes_per_graph)
                            if data.num_nodes > 0:
                                data_list.append(data)
                            else:
                                print(f"Skipping graph from file {file_path} due to zero nodes.")
                        except json.JSONDecodeError:
                            print(f"Warning: Could not decode JSON from {file_path}")
                        except Exception as e:
                            print(f"Warning: Error processing file {file_path}: {e}")
        else:
            print(f"Warning: Directory not found: {self.good_ast_dir}")
            
        if not data_list:
            print("Error: No data processed. Creating empty placeholder data.")
            # Create a placeholder to prevent torch.save from failing if data_list is empty
            placeholder_data = Data(x=torch.empty((0,3), dtype=torch.long), # 3 features
                                    edge_index=torch.empty((2,0), dtype=torch.long),
                                    edge_attr=torch.empty((0,1), dtype=torch.long),
                                    y=torch.empty((0), dtype=torch.long))
            data, slices = self.collate([placeholder_data])
        else:
            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]
            
            print("--- Before pre_transform (first 5 samples if available) ---")
            for i, d_item in enumerate(data_list[:5]): 
                original_shape = d_item.edge_index.shape if hasattr(d_item, 'edge_index') else 'N/A'
                original_num_edges = d_item.num_edges if hasattr(d_item, 'num_edges') else d_item.edge_index.size(1) if hasattr(d_item, 'edge_index') else 'N/A'
                print(f"  Original data_list[{i}].edge_index.shape: {original_shape}, num_edges: {original_num_edges}")

            if self.pre_transform is not None:
                print("--- Applying pre_transform (first 5 samples if available) ---")
                transformed_data_list = []
                for i, d_item in enumerate(data_list):
                    try:
                        # Ensure data has num_nodes if pre_transform needs it (GraphPartitionTransform might)
                        if not hasattr(d_item, 'num_nodes') or d_item.num_nodes is None:
                             d_item.num_nodes = d_item.x.size(0) if hasattr(d_item, 'x') and d_item.x is not None else 0
                        
                        transformed_d = self.pre_transform(d_item.clone()) 
                        transformed_data_list.append(transformed_d)
                        if i < 5:
                             transformed_shape = transformed_d.edge_index.shape if hasattr(transformed_d, 'edge_index') else 'N/A'
                             transformed_num_edges = transformed_d.num_edges if hasattr(transformed_d, 'num_edges') else transformed_d.edge_index.size(1) if hasattr(transformed_d, 'edge_index') else 'N/A'
                             print(f"  Transformed data_list[{i}].edge_index.shape: {transformed_shape}, num_edges: {transformed_num_edges}")
                             if hasattr(transformed_d, 'subgraphs_nodes_mapper'):
                                 print(f"    Transformed data_list[{i}] has subgraphs_nodes_mapper.")
                             else:
                                 print(f"    !!! Transformed data_list[{i}] MISSING subgraphs_nodes_mapper (This is expected if n_patches is None).")
                    except Exception as e_trans:
                        print(f"Error during pre_transform for data_list item original index {i} (file: {d_item.file_path if hasattr(d_item, 'file_path') else 'N/A'}): {e_trans}. Skipping this item.")
                data_list = transformed_data_list
            
            if not data_list:
                 print("Error: data_list became empty after pre_transform or was initially empty. Creating placeholder.")
                 placeholder_data = Data(x=torch.empty((0,3), dtype=torch.long),
                                    edge_index=torch.empty((2,0), dtype=torch.long),
                                    edge_attr=torch.empty((0,1), dtype=torch.long),
                                    y=torch.empty((0), dtype=torch.long))
                 # Ensure placeholder is compatible with pre_transform if it's expected to run on it
                 # However, at this stage, pre_transform should have already run or data_list was empty before.
                 data, slices = self.collate([placeholder_data])
            else:
                print("--- Before collate (after potential pre_transform, first 5 samples if available) ---")
                for i, d_item in enumerate(data_list[:5]):
                    collate_shape = d_item.edge_index.shape if hasattr(d_item, 'edge_index') else 'N/A'
                    collate_num_edges = d_item.num_edges if hasattr(d_item, 'num_edges') else d_item.edge_index.size(1) if hasattr(d_item, 'edge_index') else 'N/A'
                    collate_num_nodes = d_item.num_nodes if hasattr(d_item, 'num_nodes') else 'N/A'
                    print(f"  Data for collate [{i}].edge_index.shape: {collate_shape}, num_edges: {collate_num_edges}, num_nodes: {collate_num_nodes}")
                    if isinstance(d_item, SubgraphsData) or 'SubgraphsData' in str(type(d_item)):
                         print(f"    subgraphs_nodes_mapper exists: {hasattr(d_item, 'subgraphs_nodes_mapper')}")
                         if hasattr(d_item, 'subgraphs_nodes_mapper') and d_item.subgraphs_nodes_mapper is not None:
                             print(f"    subgraphs_nodes_mapper shape: {d_item.subgraphs_nodes_mapper.shape}")

                data, slices = self.collate(data_list)

        # --- Debugging Check Start ---
        print("--- After collate, before saving --- ")
        if hasattr(data, 'edge_index') and 'edge_index' in slices:
            print(f"  Collated global data.edge_index.shape: {data.edge_index.shape}")
            print(f"  slices['edge_index']: {slices['edge_index']}")
            # Print num_nodes slices as well for comparison, as Data.__inc__ for edge_index uses num_nodes
            if 'num_nodes' not in slices and hasattr(data, 'num_nodes'): # num_nodes might not be sliced if it's a Python int list
                 print(f"  Global data.num_nodes (if not sliced, likely a list of Python ints): {data.num_nodes}")
            elif 'num_nodes' in slices:
                 print(f"  slices['num_nodes']: {slices['num_nodes']}") # Should be [0, N1, N1+N2, ...]

            # Manual reconstruction check for the first few graphs from collated data
            num_graphs_to_check_manual = min(5, len(slices.get('x', [])) - 1 if 'x' in slices and len(slices.get('x', [])) > 0 else 0)
            if num_graphs_to_check_manual > 0 and data_list: # Ensure data_list is not empty
                print(f"--- Manually reconstructing edge_index for first {num_graphs_to_check_manual} graphs from collated data --- ")
                collated_edge_index_tensor = data.edge_index
                edge_index_slice_boundaries = slices['edge_index']

                for i in range(num_graphs_to_check_manual):
                    original_graph_from_list = data_list[i]
                    original_ei_shape_in_list = original_graph_from_list.edge_index.shape
                    original_ei_num_edges_in_list = original_graph_from_list.edge_index.size(1)
                    
                    start_slice = edge_index_slice_boundaries[i].item()
                    end_slice = edge_index_slice_boundaries[i+1].item()

                    print(f"  Graph {i} (from pre-collate data_list): Original edge_index shape: {original_ei_shape_in_list}, num_edges: {original_ei_num_edges_in_list}")
                    print(f"    Slicing collated_edge_index with: start_idx={start_slice}, end_idx={end_slice}")

                    reconstructed_ei = collated_edge_index_tensor[:, start_slice:end_slice]
                    print(f"    Manually reconstructed edge_index[{i}] shape: {reconstructed_ei.shape}")

                    if original_ei_num_edges_in_list == 0:
                        # For a graph that originally had 0 edges
                        if reconstructed_ei.shape[0] != 2 or reconstructed_ei.shape[1] != 0:
                            print(f"    !!!!!!!! PROBLEM: Graph {i} (0 edges) reconstructed shape is {reconstructed_ei.shape}, expected torch.Size([2,0]) !!!!!!!!")
                    else:
                        # For a graph that originally had non-zero edges
                        if reconstructed_ei.shape[0] != 2:
                            print(f"    !!!!!!!! PROBLEM: Graph {i} reconstructed edge_index first dim is {reconstructed_ei.shape[0]}, expected 2 !!!!!!!!")
                        elif reconstructed_ei.shape[1] != original_ei_num_edges_in_list:
                            print(f"    !!!!!!!! WARNING: Graph {i} reconstructed num_edges {reconstructed_ei.shape[1]} != original num_edges {original_ei_num_edges_in_list} !!!!!!!!")
            else:
                print("  Skipping manual reconstruction check (not enough graphs or data_list empty).")
        else:
            print("  Collated data does not have 'edge_index' or slices['edge_index'] is missing.")
        print("--- Debugging Check End ---")

        print(f"Saving processed data to {self.processed_paths[0]}...")
        torch.save((data, slices), self.processed_paths[0])
        
        print(f"Node type vocabulary size: {len(vocab_data['node_type_vocab'])}")
        print(f"Node name vocabulary size: {len(vocab_data['node_name_vocab'])}")
        print(f"Node value vocabulary size: {len(vocab_data['node_value_vocab'])}")
        # print("Node type vocabulary:", vocab_data['node_type_vocab'])
        # print("Node name vocabulary:", vocab_data['node_name_vocab'])
        # print("Node value vocabulary:", vocab_data['node_value_vocab'])


if __name__ == '__main__':
    bad_ast_dir_path = './deobfuscate_ast_output_bad'
    good_ast_dir_path = './deobfuscate_ast_output_good'
    dataset_root_path = './data/my_ast_dataset_16000_1_patches' # Changed root to avoid overwriting
    
    # 控制每个图的最大节点数 (例如，500 或 1000)。设置为 None 则不限制。
    MAX_NODES_PER_GRAPH = 16000
    N_PATCHES_FOR_PARTITION = 1 # Restore partitioning
    # Additional args for GraphPartitionTransform if needed, e.g., to turn off metis:
    # GRAPH_PARTITION_CUSTOM_ARGS = {'metis': False}
    GRAPH_PARTITION_CUSTOM_ARGS = {} # Use defaults for now

    print(f"Initializing dataset. Processed data will be saved in: {os.path.join(dataset_root_path, 'processed')}")
    
    dataset_instance = ASTDataset(root=dataset_root_path, 
                                  bad_ast_dir=bad_ast_dir_path, 
                                  good_ast_dir=good_ast_dir_path,
                                  max_nodes_per_graph=MAX_NODES_PER_GRAPH,
                                  n_patches_for_partition=N_PATCHES_FOR_PARTITION,
                                  graph_partition_args=GRAPH_PARTITION_CUSTOM_ARGS)
    
    print(f"Dataset created successfully!")
    print(f"Number of graphs: {len(dataset_instance)}")
    if len(dataset_instance) > 0 and dataset_instance[0].num_nodes > 0 :
        print(f"First graph: {dataset_instance[0]}")
        print(f"Number of node features in the first graph: {dataset_instance[0].num_node_features}")
        print(f"Number of nodes in the first graph: {dataset_instance[0].num_nodes}")
        print(f"Node features of the first node: {dataset_instance[0].x[0]}")
        if dataset_instance[0].num_edges > 0:
             print(f"Edge attributes of the first edge: {dataset_instance[0].edge_attr[0]}")
        else:
            print("First graph has no edges.")

    # 保存词汇表以供后续使用或分析 (可选)
    vocab_save_path = os.path.join(dataset_root_path, 'processed', 'vocabs.json')
    with open(vocab_save_path, 'w') as f:
        json.dump(vocab_data, f, indent=2)
    print(f"Vocabularies saved to {vocab_save_path}")
    
    # --- Debugging loaded data ---
    print("--- Slices from dataset_instance AFTER load ---")
    if hasattr(dataset_instance, 'slices') and dataset_instance.slices is not None and 'edge_index' in dataset_instance.slices:
        print(f"  dataset_instance.slices['edge_index']: {dataset_instance.slices['edge_index']}")
    else:
        print("  dataset_instance.slices['edge_index'] not found, not accessible, or dataset_instance.slices is None.")

    print(f"Checking dataset after loading from .pt file. Total graphs: {len(dataset_instance)}")
    num_checked = 0
    problematic_shapes_found = 0
    for i in range(min(len(dataset_instance), 32)): # 检查前32个图，或者更多
        try:
            data_item = dataset_instance[i]
            shape = data_item.edge_index.shape
            print(f"  Graph {i}: edge_index shape: {shape}")
            if shape[0] != 2: # 检查第一个维度是否为 2
                 print(f"  !!! PROBLEM: Graph {i} edge_index first dimension is NOT 2 !!!")
                 problematic_shapes_found += 1
            num_checked += 1
        except Exception as e:
            print(f"  Error accessing or checking graph {i}: {e}")

    if problematic_shapes_found > 0:
        print(f"Found {problematic_shapes_found} graphs with problematic edge_index shape immediately after loading.")
    elif num_checked > 0:
        print("Checked initial graphs, edge_index shapes seem correct immediately after loading.")
    else:
        print("Could not check any graphs after loading (dataset might be empty?).")
    print("--- End Debugging loaded data ---")
