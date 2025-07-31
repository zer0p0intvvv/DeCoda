import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn import GCNConv, GINConv

from N2C_Prop_T import Double_Level_MessageProp_random_walk_wo_norm, Double_Level_KeyProp_random_walk_wo_norm, Double_Level_MessageProp_random_walk_w_norm, Double_Level_KeyProp_random_walk_w_norm
from einops.layers.torch import Rearrange


def get_convs(args, current_input_dim):

    convs = nn.ModuleList()

    _input_dim = current_input_dim
    _output_dim = args.num_hidden

    for _ in range(args.num_convs):

        if args.conv == 'GCN':

            conv = GCNConv(_input_dim, _output_dim)

        elif args.conv == 'GIN':

            conv = GINConv(
                nn.Sequential(
                    nn.Linear(_input_dim, _output_dim),
                    nn.ReLU(),
                    nn.Linear(_output_dim, _output_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(_output_dim),
                ), train_eps=False)

        convs.append(conv)

        _input_dim = _output_dim
        _output_dim = _output_dim

    return convs


def get_input_transform(args, current_input_dim):

    return nn.Sequential(
        nn.Linear(current_input_dim, args.num_hidden),
        nn.ReLU(),
        nn.Dropout(p=args.dropout),
        nn.Linear(args.num_hidden, args.num_hidden),
        nn.ReLU(),
        nn.Dropout(p=args.dropout)
    )


def get_classifier(args):

    if args.residual == 'cat':
        return nn.Sequential(
            nn.Linear(args.num_hidden*2, args.num_hidden),
            nn.ReLU(),
            nn.Dropout(p=args.dropout),
            nn.Linear(args.num_hidden, args.num_classes)
        )
    else:
        return nn.Sequential(
            nn.Linear(args.num_hidden, args.num_hidden//2),
            nn.ReLU(),
            nn.Dropout(p=args.dropout),
            nn.Linear(args.num_hidden//2, args.num_classes)
        )


def get_deepset_layer(args, input_dim, output_dim, num_layers):
    layers = []
    if num_layers == 1:
        layers.append(nn.Linear(input_dim, output_dim))
    else:
        layers.append(nn.Linear(input_dim, args.num_hidden))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=args.dropout))
        for _ in range(1, num_layers-1):
            layers.append(nn.Linear(args.num_hidden, args.num_hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=args.dropout))
        layers.append(nn.Linear(args.num_hidden, output_dim))
    if args.layernorm:
        layers.append(nn.LayerNorm(output_dim))
    return nn.Sequential(*layers)


class Cluster_GT(torch.nn.Module):

    def __init__(self, args):

        super(Cluster_GT, self).__init__()

        self.args = args
        self.use_rw = args.pos_enc_rw_dim > 0
        self.use_lap = args.pos_enc_lap_dim > 0
        self.num_classes = args.num_classes
        self.nhid = args.num_hidden
        self.attention_based_readout = args.attention_based_readout
        self.prop_w_norm_on_coarsened = args.prop_w_norm_on_coarsened
        self.residual = args.residual
        self.remain_k1 = args.remain_k1
        self.d_k_tensor = torch.tensor(args.num_hidden).float()

        self.kernel_method = args.kernel_method
        assert self.kernel_method in ['relu', 'elu']

        self.is_ast_dataset = (args.data == 'AST_MALWARE')
        self.total_embedding_dim = 0

        current_feature_dim_for_input_layers = args.num_features # Default for TUDataset like data

        if self.is_ast_dataset:
            if args.node_type_vocab_size is None or args.node_name_vocab_size is None or args.node_value_vocab_size is None:
                raise ValueError("Vocabulary sizes must be provided for AST_MALWARE dataset via args.")
            
            self.type_embedding = nn.Embedding(args.node_type_vocab_size, args.embedding_dim_type)
            self.name_embedding = nn.Embedding(args.node_name_vocab_size, args.embedding_dim_name)
            self.value_embedding = nn.Embedding(args.node_value_vocab_size, args.embedding_dim_value)
            self.total_embedding_dim = args.embedding_dim_type + args.embedding_dim_name + args.embedding_dim_value
            current_feature_dim_for_input_layers = self.total_embedding_dim
        
        # Add positional encoding dimensions to the feature dimension for GNN/input_transform
        effective_input_dim = current_feature_dim_for_input_layers
        # For AST_MALWARE, ast_to_pyg.py currently does NOT add positional encodings.
        # So, only add PE dims if it's NOT AST_MALWARE, or if AST_MALWARE is later updated to include them.
        if not self.is_ast_dataset: 
            if self.use_rw:
                effective_input_dim += args.pos_enc_rw_dim
            if self.use_lap:
                effective_input_dim += args.pos_enc_lap_dim
        # If you later modify ast_to_pyg.py to include PEs for AST_MALWARE and want to use them,
        # you would remove "if not self.is_ast_dataset:" condition above.
        # And ensure that data object in forward() actually has rw_pos_enc/lap_pos_enc.

        if args.use_gnn:
            self.convs = get_convs(args, effective_input_dim)
        else:
            self.input_transform = get_input_transform(args, effective_input_dim)

        if args.attention_based_readout:
            self.readout_seed_vector = nn.Parameter(
                torch.randn(args.num_hidden), requires_grad=True)

        self.classifier = get_classifier(args)

        self.subgraph_combined_pre_deepset = get_deepset_layer(
            args, args.num_hidden, args.num_hidden, args.deepset_layers)
        self.subgraph_combined_post_deepset = get_deepset_layer(
            args, args.num_hidden, args.num_hidden, args.deepset_layers)

        self.diffQ = args.diffQ
        if args.diffQ:
            self.subgraph_combined_pre_deepset_prime = get_deepset_layer(
                args, args.num_hidden, args.num_hidden, args.deepset_layers)
            self.subgraph_combined_post_deepset_prime = get_deepset_layer(
                args, args.num_hidden, args.num_hidden, args.deepset_layers)

        self.subgraph_combined_linK = nn.Linear(
            args.num_hidden, args.num_hidden)
        self.subgraph_combined_linV = nn.Linear(
            args.num_hidden, args.num_hidden)

        self.patch_rw_dim = args.pos_enc_patch_rw_dim
        if self.patch_rw_dim > 0:
            if self.residual == 'cat':
                self.patch_rw_encoder = nn.Linear(
                    self.patch_rw_dim, 2 * args.num_hidden)
            else:
                self.patch_rw_encoder = nn.Linear(
                    self.patch_rw_dim, args.num_hidden)

        if args.prop_w_norm_on_coarsened:
            self.propM = Double_Level_MessageProp_random_walk_wo_norm(
                node_dim=-3)
            self.propK = Double_Level_KeyProp_random_walk_wo_norm(node_dim=-2)
        else:
            self.propM = Double_Level_MessageProp_random_walk_w_norm(
                node_dim=-3)
            self.propK = Double_Level_KeyProp_random_walk_w_norm(node_dim=-2)

        self.reshape = Rearrange('(B p) d ->  B p d', p=args.n_patches)

        self.layernorm_flag = args.layernorm
        if self.layernorm_flag:
            self.layernorm_gnn_input = nn.LayerNorm(args.num_hidden)

    def forward(self, data):
        if self.is_ast_dataset:
            if not (hasattr(data, 'x') and data.x is not None and data.x.size(1) == 3):
                 raise ValueError("AST_MALWARE data must have data.x with 3 features (type_id, name_id, value_id).")
            type_ids = data.x[:, 0]
            name_ids = data.x[:, 1]
            value_ids = data.x[:, 2]

            type_embeds = self.type_embedding(type_ids)
            name_embeds = self.name_embedding(name_ids)
            value_embeds = self.value_embedding(value_ids)
            
            current_x = torch.cat([type_embeds, name_embeds, value_embeds], dim=-1)
        else:
            current_x = data.x

        # node PE
        # For AST_MALWARE, only concatenate PEs if ast_to_pyg.py was modified to add them AND
        # the effective_input_dim in __init__ was also calculated to include them.
        # Given the change above in __init__ (PE dims not added to effective_input_dim for AST_MALWARE),
        # we should also avoid concatenating them here for AST_MALWARE to maintain consistency,
        # unless the data object for AST_MALWARE is explicitly prepared with these PEs by ast_to_pyg.py
        # and __init__ logic is reverted.
        if not self.is_ast_dataset: # Only apply for non-AST datasets as per current __init__ logic for effective_input_dim
            if self.use_rw and hasattr(data, 'rw_pos_enc'):
                current_x = torch.cat((current_x, data.rw_pos_enc), dim=-1)
            if self.use_lap and hasattr(data, 'lap_pos_enc'):
                current_x = torch.cat((current_x, data.lap_pos_enc), dim=-1)
        elif self.is_ast_dataset: # If it is AST_MALWARE, ensure current_x is the embedded features
                                  # and PEs are only added if ast_to_pyg.py includes them AND init logic accounts for them.
                                  # Current init logic for AST_MALWARE sets effective_input_dim = total_embedding_dim.
                                  # So, if PEs were present on data object for AST, they should NOT be cat here unless init changes.
            pass # current_x is already set from embeddings. PEs are not added to it for AST data based on current __init__.

        # input transform
        if self.args.use_gnn:
            for i in range(self.args.num_convs):
                current_x = F.relu(self.convs[i](current_x, data.edge_index))
                if self.layernorm_flag:
                    current_x = self.layernorm_gnn_input(current_x)
        else:
            current_x = self.input_transform(current_x)

        # get subgraph-level data
        subgraph_combined_x = current_x[data.subgraphs_nodes_mapper]
        subgraph_combined_batch = data.subgraphs_batch

        # get edges of coarsened graph
        subgraphs_batch_row = data.subgraphs_batch_row
        subgraphs_batch_col = data.subgraphs_batch_col
        coarsen_edge_attr = data.coarsen_edge_attr
        coarsen_edge_index = torch.stack(
            [data.subgraphs_batch_row, data.subgraphs_batch_col], dim=0)

        if not self.prop_w_norm_on_coarsened:
            # compute laplacian of coarsened graph
            coarsen_deg = torch.bincount(
                subgraphs_batch_row, coarsen_edge_attr)
            coarsen_deg_inv_sqrt = coarsen_deg.pow(-0.5)
            coarsen_deg_inv_sqrt[coarsen_deg_inv_sqrt == float('inf')] = 0
            coarsen_edge_attr = coarsen_deg_inv_sqrt[subgraphs_batch_row] * \
                coarsen_edge_attr * coarsen_deg_inv_sqrt[subgraphs_batch_col]

        # compute query
        subgraph_combined_Q = self.subgraph_combined_pre_deepset(
            subgraph_combined_x)
        scattered_subgraph_combined_Q = scatter(
            subgraph_combined_Q, subgraph_combined_batch, dim=0, reduce="sum")
        scattered_subgraph_combined_Q = self.subgraph_combined_post_deepset(
            scattered_subgraph_combined_Q)

        if self.diffQ:
            # compute query prime
            subgraph_combined_Q_prime = self.subgraph_combined_pre_deepset_prime(
                subgraph_combined_x)
            scattered_subgraph_combined_Q_prime = scatter(
                subgraph_combined_Q_prime, subgraph_combined_batch, dim=0, reduce="sum")
            scattered_subgraph_combined_Q_prime = self.subgraph_combined_post_deepset_prime(
                scattered_subgraph_combined_Q_prime)

        # compute key and value
        subgraph_combined_K = self.subgraph_combined_linK(subgraph_combined_x)
        scattered_subgraph_combined_K = scatter(
            subgraph_combined_K, subgraph_combined_batch, dim=0, reduce="mean")
        subgraph_combined_V = self.subgraph_combined_linV(subgraph_combined_x)

        # kernelized
        if self.kernel_method == 'relu':
            kernelized_scattered_subgraph_combined_Q = F.relu(
                scattered_subgraph_combined_Q)
            kernelized_subgraph_combined_K = F.relu(subgraph_combined_K)
            kernelized_scattered_subgraph_combined_K = F.relu(
                scattered_subgraph_combined_K)
            if self.diffQ:
                kernelized_scattered_subgraph_combined_Q_prime = F.relu(
                    scattered_subgraph_combined_Q_prime)
        elif self.kernel_method == 'elu':
            kernelized_scattered_subgraph_combined_Q = 1 + \
                F.elu(scattered_subgraph_combined_Q)
            kernelized_subgraph_combined_K = 1 + F.elu(subgraph_combined_K)
            kernelized_scattered_subgraph_combined_K = 1 + \
                F.elu(scattered_subgraph_combined_K)
            if self.diffQ:
                kernelized_scattered_subgraph_combined_Q_prime = 1 + \
                    F.elu(scattered_subgraph_combined_Q_prime)

        # compute double-level (subgraph-wise) qv gate
        if self.remain_k1:
            subgraph_qv_gate = torch.exp(
                (scattered_subgraph_combined_K[subgraphs_batch_row] * scattered_subgraph_combined_Q[subgraphs_batch_col]/torch.sqrt(self.d_k_tensor)).sum(dim=1, keepdim=True))
        else:
            subgraph_qv_gate = (kernelized_scattered_subgraph_combined_K[subgraphs_batch_row] *
                                kernelized_scattered_subgraph_combined_Q[subgraphs_batch_col]).sum(dim=1, keepdim=True)

        # scatter kernelized K
        scattered_kernelized_subgraph_combined_K = scatter(
            kernelized_subgraph_combined_K, subgraph_combined_batch, dim=0, reduce="sum")

        # compute message and scatter kernelized message
        kernelized_subgraph_combined_M = torch.einsum(
            'ni,nj->nij', [kernelized_subgraph_combined_K, subgraph_combined_V])
        scattered_kernelized_subgraph_combined_M = scatter(
            kernelized_subgraph_combined_M, subgraph_combined_batch, dim=0, reduce="sum")

        # propagate message and key on the coarsened graph
        if not self.prop_w_norm_on_coarsened:
            scattered_kernelized_subgraph_combined_M = self.propM(
                scattered_kernelized_subgraph_combined_M, coarsen_edge_index, coarsen_edge_attr.view(-1, 1, 1), subgraph_qv_gate.view(-1, 1, 1))
            scattered_kernelized_subgraph_combined_K = self.propK(
                scattered_kernelized_subgraph_combined_K, coarsen_edge_index, coarsen_edge_attr.view(-1, 1), subgraph_qv_gate.view(-1, 1))
        else:
            scattered_kernelized_subgraph_combined_M = self.propM(
                scattered_kernelized_subgraph_combined_M, coarsen_edge_index, subgraph_qv_gate.view(-1, 1, 1))
            scattered_kernelized_subgraph_combined_K = self.propK(
                scattered_kernelized_subgraph_combined_K, coarsen_edge_index, subgraph_qv_gate.view(-1, 1))

        # compute attention
        if self.diffQ:
            kernelized_subgraph_combined_H = torch.einsum(
                'ni,nij->nj', [kernelized_scattered_subgraph_combined_Q_prime, scattered_kernelized_subgraph_combined_M])
            kernelized_subgraph_combined_C = torch.einsum(
                'ni,ni->n', [kernelized_scattered_subgraph_combined_Q_prime, scattered_kernelized_subgraph_combined_K]).unsqueeze(-1) + 1e-6
        else:
            kernelized_subgraph_combined_H = torch.einsum(
                'ni,nij->nj', [kernelized_scattered_subgraph_combined_Q, scattered_kernelized_subgraph_combined_M])
            kernelized_subgraph_combined_C = torch.einsum(
                'ni,ni->n', [kernelized_scattered_subgraph_combined_Q, scattered_kernelized_subgraph_combined_K]).unsqueeze(-1) + 1e-6
        out = kernelized_subgraph_combined_H / kernelized_subgraph_combined_C

        # residual connection
        if self.residual in ['sum', 'cat']:
            scattered_subgraph_combined_x = scatter(
                subgraph_combined_x, subgraph_combined_batch, dim=0, reduce="mean")
            if self.residual == 'sum':
                out = out + scattered_subgraph_combined_x
            elif self.residual == 'cat':
                out = torch.cat((out, scattered_subgraph_combined_x), dim=-1)

        # Patch PE
        if self.patch_rw_dim > 0:
            out += self.patch_rw_encoder(data.patch_pe)

        # reshape from (number of patches of the whole batch, hidden_dim) to (graph_id, number of patches, hidden_dim)
        out = self.reshape(out)

        # attention-based readout
        if self.attention_based_readout:
            inner_products = torch.einsum(
                'ijk,k->ij', out, self.readout_seed_vector)
            readout_attention_weights = F.softmax(inner_products, dim=-1)
            out = torch.einsum('ij,ijk->ik', readout_attention_weights, out)
        # average pooling
        else:
            out = (out * data.mask.unsqueeze(-1)).sum(1) / \
                data.mask.sum(1, keepdim=True)

        # output decoder
        out = self.classifier(out)

        return F.log_softmax(out, dim=-1)

