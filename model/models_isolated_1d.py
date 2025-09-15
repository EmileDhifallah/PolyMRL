from torch import nn
import torch
import math
import os

# dynamics imports
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import root_mean_squared_error, r2_score

from model.egnn_new import EGNN, GNN
from helpers_from_edm.utils import remove_mean, remove_mean_with_mask, assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked, sample_center_gravity_zero_gaussian_with_mask, sample_gaussian_with_mask
from helpers_from_edm.en_diffusion import GammaNetwork

import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback, AutoConfig, AutoModelForMaskedLM
from transformers import RobertaConfig, RobertaTokenizerFast, RobertaForMaskedLM, DataCollatorForLanguageModeling # for ChemBERTa
from transformers import OpenAIGPTForSequenceClassification, DataCollatorWithPadding


class EGNN_dynamics_QM9(nn.Module):
    def __init__(self, in_node_nf, context_node_nf,
                 n_dims, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 condition_time=False, tanh=False, mode='egnn_dynamics', norm_constant=0,
                 inv_sublayers=2, sin_embedding=False, normalization_factor=100, 
                 aggregation_method='sum', reflection_equiv=True):
        super().__init__()

        print("EGNN dynamics: in_node_nf, context_node_nf,n_dims, hidden_nf=64, device=cpu, act_fn=torch.nn.SiLU(), n_layers=4, attention=False,condition_time=True")
        print(in_node_nf, context_node_nf, # 6 0 3 64, n_layers=4
                 n_dims, hidden_nf, device,
                 act_fn, n_layers, attention,
                 condition_time)

        self.mode = mode
        if mode == 'egnn_dynamics':
            self.egnn = EGNN(
                in_node_nf=in_node_nf + context_node_nf, in_edge_nf=1, # in edge nf 1 model_dynamics
                hidden_nf=hidden_nf, device=device, act_fn=act_fn,
                n_layers=n_layers, attention=attention, tanh=tanh, norm_constant=norm_constant,
                inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                reflection_equiv=reflection_equiv)
            print('egnn instance params: in_node_nf + context_node_nf,hidden_nf', in_node_nf + context_node_nf,hidden_nf)

            self.in_node_nf = in_node_nf
        elif mode == 'gnn_dynamics':
            self.gnn = GNN(
                in_node_nf=in_node_nf + context_node_nf + 3, in_edge_nf=0,
                hidden_nf=hidden_nf, out_node_nf=3 + in_node_nf, device=device,
                act_fn=act_fn, n_layers=n_layers, attention=attention,
                normalization_factor=normalization_factor, aggregation_method=aggregation_method)

        self.context_node_nf = context_node_nf
        self.device = device
        self.n_dims = n_dims
        print('n_dims: ', n_dims)
        self._edges_dict = {}
        self.condition_time = condition_time

    # def forward(self, t, xh, node_mask, edge_mask, context=None):
    #     raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)
        return fwd

    def unwrap_forward(self):
        return self._forward

    ### warning: we renamed '_forward' to 'forward'!
    def forward(self, t, xh, node_mask, edge_mask, context):
        '''
        t: timestep, scalar
        xh: x and h concatenated, (batch, seq_len, 6+3) = (64, 25, 9);
        x here is the noised set of coordinates, of (batch, seq_len, 3)
        
        '''
        
        bs, n_nodes, dims = xh.shape

        h_dims = dims - self.n_dims
        # print('h_dims: ', h_dims) # 6, = len of vocab, i.e. nr. of possible atoms in dataset

        # build (fully-connected?) adjacency matrix
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [x.to(self.device) for x in edges] # (40000, ), (40000, )

        edges_0 = edges[0].tolist()
        edges_1 = edges[1].tolist()
        # print('edges length: ', len(edges_0), len(edges_1))
        # print('edges: ')

        # for i,edge in enumerate(zip(edges_0, edges_1)):
        #     print(edge)
        #     if i%(n_nodes**2-1)==0:
        #         print(f'molecule of size {n_nodes}x{n_nodes} passed')


        # print('edges: ', [i for i in edges[0]], [i for i in edges[1]])
        # print('edges 0: ', edges[0].shape)

        node_mask = node_mask.view(bs*n_nodes, 1)
        edge_mask = edge_mask.view(bs*n_nodes*n_nodes, 1)
        # print('nodemask edgemask reordered: ', node_mask.shape, edge_mask.shape)
        # (1600, 1), (40000, 1)

        # multiply xh with the node_mask after reorder
        xh = xh.view(bs*n_nodes, -1).clone() * node_mask # (1600, 9)


        # separate x and h from the concatenation xh
        x = xh[:, 0:self.n_dims].clone()
        if h_dims == 0:
            h = torch.ones(bs*n_nodes, 1).to(self.device)
        else:
            h = xh[:, self.n_dims:].clone()
        # so now: x = (64, 25, 3) and h = (64, 25,6)


        ### remove time condition [in non-diffusion variant] (or just leave it and set it to False?)
        # concat timestep t to h in case we condition on timestep t
        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t.view(bs, 1).repeat(1, n_nodes)
                h_time = h_time.view(bs * n_nodes, 1)
            # print('h time: ', h_time)
            h = torch.cat([h, h_time], dim=1)
            # print('h plus h time: ', h.shape) # (1600, 7)

        # if we use property condition, concat this to h also on last dim., so we get h=(batch*seq_len, vocab_size+1(for time) + cond_num_feat)
        if context is not None:
            # We're conditioning, awesome!
            context = context.view(bs*n_nodes, self.context_node_nf)
            # print('context: ', context)
            h = torch.cat([h, context], dim=1)
            # print('h plus context: ', h.shape)

        # run the actual EGNN on the input node types/featurs, coordinates, edge lists, node and edge masks
        if self.mode == 'egnn_dynamics':
            h_final, x_final, h_hidden = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask)
            # print('h final: ', h_final.shape)
            # print('x final: ', x_final.shape)
            # note: these are x_pred(^) and 'eps_h'


            ### now, alternatively; could we remove the vel and use the predicted x instead of the noise prediction in the L2 loss? Try both.
            # the velocity ('momentum'?) is the final coordinate representation minus the initial coordinate representation (of the not-masked atoms)
            # note: x_final is clean predicted coordinate, x is input noisy coordinate;
            # the difference is the predicted coordinate noise 'eps'
            # note: vel = 'eps_x':
            vel = (x_final - x) * node_mask  # This masking operation is redundant but just in case

        elif self.mode == 'gnn_dynamics':
            xh = torch.cat([x, h], dim=1)
            output = self.gnn(xh, edges, node_mask=node_mask)
            vel = output[:, 0:3] * node_mask
            h_final = output[:, 3:]

        else:
            raise Exception("Wrong mode %s" % self.mode)

        if context is not None:
            # Slice off context size: --> we learned context; now we remove it from the representation! --> WHY?
            h_final = h_final[:, :-self.context_node_nf]

        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :-1]

        vel = vel.view(bs, n_nodes, -1) # from (1600, 3) to (64, 25, 3)

        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan, resetting EGNN output to zero.')
            vel = torch.zeros_like(vel)

        ### think we need to keep these
        if node_mask is None:
            vel = remove_mean(vel)
        else:
            vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))

        if h_dims == 0:
            return vel
        # return the velocity (coordinate representation 'update'), together with the final node feature (=atom type) representation,
        # concatenated into one tensor --> this now goes back into the diffusion wrapper forward; is this supposed to be the predicted noise
        # out of the full input representation/coordinates+node types?
        else:
            # back from (batch*seq_len, h_num_feat) to (batch, seq_len, h_num_feat); from (1600, 6) to (64, 25, 6)
            h_final = h_final.view(bs, n_nodes, -1) # = (64, 25, 6); not sure why the time dimension (+1) is not in here; copilot says it is not needed for the diffusion model?
            h_hidden = h_final.view(bs, n_nodes, -1)
            
            # concatenate eps_h and eps_x into eps_x+h
            return torch.cat([vel, h_final], dim=2), h_final, h_hidden

    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [torch.LongTensor(rows).to(device),
                         torch.LongTensor(cols).to(device)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)

### step after dynamics: compute L2 with net_output and input_xh (noised input)



### The EGNN model
class GCL(nn.Module):
    '''
    Class for setting up and computing outputs of a graph convolutional layer on the node representations of
    the input molecule -> representation 'h' in the paper.
    '''

    def __init__(self, input_nf, output_nf, hidden_nf, normalization_factor, aggregation_method, # I believe input_nf is the number of atoms in the molecule
                 edges_in_d=0, nodes_att_dim=0, act_fn=nn.SiLU(), attention=False):
        super(GCL, self).__init__()

        # not sure what input_nf is
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention # whether to use attention scores on top of edge message reprs. (multiplication w/ attn. score)

        # print('edge hiddennf: ', hidden_nf)
        # print('input_edge + edges_in_d: ', input_edge, edges_in_d)
        # MLP from some input dimensionality to hidden dimens., to encode input edge representations+features
        self.edge_mlp = nn.Sequential(
            # input dim.: edge feature length + number of edges to analyse (for us, every edge in the molecule, I believe)
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf), # map to hidden dim.
            act_fn)
        
        # MLP from some input dimensionality to hidden dimens., to encode input node representations+features
        self.node_mlp = nn.Sequential(
            # input dim.: node input feature length + hidden feature length + node feature input emb. dimensionality
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf)) # map to output dim.

        # MLP to compute attention score; takes computed edge message representation,
        # and literally just maps this representation down to 1 scalar value, the attention score for this specific
        # message between nodes i and j
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, edge_attr, edge_mask):
        '''
        Computing the representation of the edge going from node 'source' to node 'target',
        which is essentially the message/importance of the edge (i,j) (or (source, target)) for the representation
        of node 'source'/i. IN this computation:
        
        - We run the representations of the nodes source and target through the edge-focussed MLP defined above, by
            concatenating the source and target reprs., and concatenating the representation of this edge's features/attributes
            ('edge_attr') to this also -> this concatenated tensor goes into the edge mlp creating the message m_ij;
        - the attention score for this respective message, m_ij, is computed by running the message repr. through the
            attention-focussed MLP -> the resulting attention score is multiplied with the message representation itself;
        - if we provide a mask for this edge, for example because certain edges we do not want to count towards
            a node's representation, then multiply that mask with the message representation, either making it 0
            or keeping it just the same (x 1);
        - we then return the message representation 'out', as well as the message representation before applying
            attention and the edge mask (as it came out of the edge mlp), 'm_ij'.

        '''
        if edge_attr is None:  # Unused.
            # print('source sh: ', source.shape)
            # print('target sh: ', target.shape)
            out = torch.cat([source, target], dim=1)
            
        else:
            # print('source sh: ', source.shape)
            # print('target sh: ', target.shape)
            # print('edge_attr sh: ', edge_attr.shape)
            out = torch.cat([source, target, edge_attr], dim=1)
        mij = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        if edge_mask is not None:
            out = out * edge_mask
        return out, mij

    def node_model(self, x, edge_index, edge_attr, node_attr):
        '''
        Computes update to the atoms node reprs. h and returns new coordinates h(l+1);

        inputs:
        x: node reprs., i.e. h, for every atom i in h / the molecule;
        edge_index: I believe an adjacency matrix represented as 2 lists, indicating every edge that exists in
        the molecule, i.e. every position where there is a bond between 2 atoms (i,j);
        edge_attr: for every (existing (I think?) edge, the edge feature representation
        node_attr: node feature representation for every atom i in h/the molecule;
        '''
        
        row, col = edge_index # take the row-part of the adjacency matrix (not sure what this looks like but it could be that row and col are identical?);
        # this will serve as the segment_ids (2nd arg in unsorted_segment_sum) for dividing+summing the edge feature reprs.

        # sum-aggregation of all the edge feature representations, for all the edges that exist in the molecule
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        
        # concatenate the node reprs. with the summed-up edge feature reprs. ('agg') and the node feature reprs.
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        
        # now run the concatenated repr. through the node-focussed MLP; the output, of dim. 'output_dim', has the same size
        # as x, which means the nr. of atoms in the molecule; now we sum this output onto the original node reprs. x (i.e. 'h' in paper);
        out = x + self.node_mlp(agg) # this gives the node-repr.-update
        # return the new version of x ('out'), together with the concatenated represenation of x, edge and node features (called 'agg')
        return out, agg

    def forward(self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        '''
        the forward of the graph convolution layer; computes updates to all the edges and nodes in the graph/molecule,
        and returns the updated node representations, h(l+1), and the computed messages between atoms (i,j).
        '''

        # get row/col of the adjacency matrix of edges; this indicates which edges exist (like a binary mask)
        row, col = edge_index
        # get updated edge representations, input to edge MLP are the node representations of the atoms that exist (acquired using existing edge indication in row and col),
        # as well as the edge feature reprs. and the edge mask (only if given as input)
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        # enter the full node reprs. h, as well as the edge adjacency matrix, the updated edge reprs. and the
        # node feature reprs. into the node-focussed MLP and get the updated node reprs. out
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        
        # apply node mask, if give as input, to make certain node reprs. that we do not want to update = 0
        if node_mask is not None:
            h = h * node_mask
        
        # now return the updated node reprs. and the message representations of the edges m_ij
        return h, mij


class EquivariantUpdate(nn.Module):
    '''
    Class for updating the atom coordinate representations.
    '''

    def __init__(self, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=1, act_fn=nn.SiLU(), tanh=False, coords_range=10.0,
                 reflection_equiv=True):
        super(EquivariantUpdate, self).__init__()

        self.tanh = tanh
        self.coords_range = coords_range
        self.reflection_equiv = reflection_equiv # does True mean we wánt reflection insensitivity (=equivariance)?

        input_edge = hidden_nf * 2 + edges_in_d # same input dim. as edge MLP in the GCN class

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # self.coord_mlp.apply(init_weights)

        # added to initialize coord mlp's layer differently:
        layer_mlp0 = nn.Linear(input_edge, hidden_nf)
        nn.init.xavier_uniform_(layer_mlp0.weight)
        nn.init.zeros_(layer_mlp0.bias)

        # fully connected layer projecting from hidden dim. down to 1 scalar; for Xavier initialization?
        layer = nn.Linear(hidden_nf, 1, bias=False)
        
        nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu')) # gain=0.001)
        # nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')

        # define an MLP to process coordinates and return new coordinate representation
        self.coord_mlp = nn.Sequential(
            layer_mlp0, #nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer)
        
        # MLP for removing reflection-equivariance (eq. 7 in DiffSBDD paper)
        self.cross_product_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer
        ) if not self.reflection_equiv else None
        
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
    
    def print_diagnostics(self, tens_name, inp_tens):
        with torch.no_grad():
            print(f"{tens_name} stats:")
            print("  norm:", inp_tens.norm().item())
            print("  min:", inp_tens.min().item())
            print("  max:", inp_tens.max().item())
            print("  mean:", inp_tens.mean().item())
            print("  std:", inp_tens.std().item())

    def coord_model(self, h, coord, edge_index, coord_diff, coord_cross,
                    edge_attr, edge_mask, update_coords_mask=None): # 'update_coords_mask' is added to the DiffSBDD-EGNN version
        '''
        Applying the coord MLP to the coordinates
        inputs;
            h: node representations for all atoms in molecule;
            coord: coordinates/coord. reprs. of all atoms in molecule;
            edge_index: adjacency matrix indicating which edges exist;
            coord_diff: relative distance d_ij between atoms i and j;
            coord_cross: cross product between coordinates (used in refl. equiv. MLP);
            edge_attr: edge feature reprs.;
            edge_mask: binary mask indicating if we want to leave some edge reprs. untouched;
            update_coords_mask: binary mask indicating if we should leave certain atoms' coordinates/coord. repr. untouched.
        '''

        # specifying which edges exist with masks row and col; row indicates nodes i and
        row, col = edge_index # col indicates nodes j for all edges/bonds (i,j) in molecule
        # print('row, col', row, col)
        # concatenate node representations of nodes i and nodes j, together with edge feature reprs.
        # print('hrow, hcol', h[row], h[col])
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        # self.print_diagnostics(/'h[row]', h[row])
        # self.print_diagnostics('h[col]', h[col])
        # self.print_diagnostics('edge_attr', edge_attr)
        
        # apply the coordinate-MLP to the concatenated input; if indicated, also apply Tanh act.
        # and multiply with given coordinate range; the update to the coord. reprs. is called 'trans'
        # print("coord_mlp input: ", x.norm(), x.min(), x.max(), x.mean())

        if self.tanh:
            print('tanh yes')
            print('coords_range: ', coords_range)

            interm = self.coord_mlp(input_tensor)
            # self.print_diagnostics('interm', interm)

            tanh = torch.tanh(interm)
            # self.print_diagnostics('tanh', tanh)

            # self.print_diagnostics('coord_diff', coord_diff)

            trans = coord_diff * tanh * self.coords_range
            # self.print_diagnostics('trans', trans)

            
        else:
            print('tanh no')
            interm=self.coord_mlp(input_tensor)
            # self.print_diagnostics('interm', interm)

            # self.print_diagnostics('coord_diff', coord_diff)

            trans = coord_diff * interm
            # self.print_diagnostics('trans', trans)

        
        # add refl. equiv. MLP if indicated, and if so, add the output of this MLP ('phi_cross') to updated coord reprs. 'trans';
        # also apply Tanh if indicated in input
        if not self.reflection_equiv:
            print('relfection equivariance no (crossproductmlp yes!)')
            phi_cross = self.cross_product_mlp(input_tensor)
            if self.tanh:
                phi_cross = torch.tanh(phi_cross) * self.coords_range
            trans = trans + coord_cross * phi_cross # coord_cross is a cross product on atom coordinates, as defined in function below

        # if we have atoms that form bonds, for which we do not want updates,
        # here we apply the edge mask (binary mask) to make sure those atoms do not get coordinate updates
        if edge_mask is not None:
            trans = trans * edge_mask

        # sum-aggregation of all calculated coordinate representations, of all existing atoms
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)

        # note: this update mask is added in the DiffSBDD-EGNN implementation
        # if we do not want to update the coordinates at all, here an optional coord. mask is applied if given
        if update_coords_mask is not None:
            print('update coords mask yes')
            # self.print_diagnostics('update_coords_mask', update_coords_mask)

            # self.print_diagnostics('agg_1', agg)
            
            agg = update_coords_mask * agg
            # self.print_diagnostics('agg_2', agg)

        # -> I think this either gives 0 everywhere for the update; if not provided the coord. update stays as is
        
        # add the coordinate update to the input coordinates, and return the new coordinates
        coord = coord + agg
        # self.print_diagnostics('coord_new', coord)

        return coord

    def forward(self, h, coord, edge_index, coord_diff, coord_cross,
                edge_attr=None, node_mask=None, edge_mask=None,
                update_coords_mask=None):
        '''
        Forward of the equivariant coordinate-update module (is this module considered as message-passing, or simply
        an equivariant MLP module?); the output of the module is the updated set of coordinates x(l+1)
        for all atoms i in the molecule.
        '''
        # get coordinate update
        coord = self.coord_model(h, coord, edge_index, coord_diff, coord_cross,
                                 edge_attr, edge_mask,
                                 update_coords_mask=update_coords_mask)
        
        # if we do not want to update certain atoms in the molecule, we apply a node mask (optional input) here
        # to the resulting, updated atom coordinates; weird thing is that, if this is a binary mask, some atoms
        # are going to have a '0' as their coordinates after mask application; why is this?
        if node_mask is not None:
            coord = coord * node_mask
        return coord


class EquivariantBlock(nn.Module):
    '''
    This module combines the atom-type update step module, GCN(), and the coordinate update step module, EquivariantUpdate(),
    and runs the atom-type module for the pre-determined number of layers (L), and the coordinate module just once,
    then outputs both updated representations, h(L), and x(L).
    '''

    def __init__(self, hidden_nf, edge_feat_nf=2, device='cpu', act_fn=nn.SiLU(), n_layers=2, attention=True,
                 norm_diff=True, tanh=False, coords_range=15, norm_constant=1, sin_embedding=None,
                 normalization_factor=100, aggregation_method='sum', reflection_equiv=True):
        super(EquivariantBlock, self).__init__()

        self.hidden_nf = hidden_nf # hidden dim.
        self.device = device
        self.n_layers = n_layers # = L
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.reflection_equiv = reflection_equiv

        # build #L GCN layers into a sequence of layers/modules (add_module is a method of nn.Module)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=edge_feat_nf, # edge_feat_nf/edges_in_d either 2 or 3 (depending on definition in EGNN!)
                                              act_fn=act_fn, attention=attention,
                                              normalization_factor=self.normalization_factor,
                                              aggregation_method=self.aggregation_method))
        # add the coordinate module too
        self.add_module("gcl_equiv", EquivariantUpdate(hidden_nf, edges_in_d=edge_feat_nf, act_fn=nn.SiLU(), tanh=tanh,
                                                       coords_range=self.coords_range_layer,
                                                       normalization_factor=self.normalization_factor,
                                                       aggregation_method=self.aggregation_method,
                                                       reflection_equiv=self.reflection_equiv))
        self.to(self.device) # to GPU (if GPU avail)

    def print_diagnostics(self, tens_name, inp_tens):
        with torch.no_grad():
            print(f"{tens_name} stats:")
            print("  norm:", inp_tens.norm().item())
            print("  min:", inp_tens.min().item())
            print("  max:", inp_tens.max().item())
            print("  mean:", inp_tens.mean().item())
            print("  std:", inp_tens.std().item())

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None,
                edge_attr=None, update_coords_mask=None, batch_mask=None): # note: batch mask added in DiffSBDD-implementation
        '''
        forward, which calls the needed modules with given inputs;

            h: initial atom-type node representations;
            x: initial atom coordinates;
            edge_index: adjacency matrix indicating existence of edges between atoms;
            node_mask: atom types that we do not want to update for;
            edge_mask: edges representations that we do not want to update for --> how is this implemented during the update equation?;
            edge_attr: optional edge features;
            update_coords_mask: atom coordinates that we do not want to update during training;
            batch_mask: not sure what this is or does.
        '''
        # Edit Emiel: Remove velocity as input

        # compute relative distances between all atoms where bonds/edges exist
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant) # distances = (33k, 1)
        # self.print_diagnostics('distances', distances)
        # self.print_diagnostics('coord_diff', coord_diff)

        # if no refl. equiv. is desired, add cross-atom coordinates for input into cross-coordinate-MLP within the coordinate MLP
        if self.reflection_equiv:
            coord_cross = None
        else:
            coord_cross = coord2cross(x, edge_index, node_mask,#batch_mask,
                                      self.norm_constant)
        
        # sinusoidal embedding module; is this a positional embedding perhaps? It's added to the edge features
        if self.sin_embedding is not None:
            print('sin_embedding yes')
            distances = self.sin_embedding(distances)

        # self.print_diagnostics('edge_attr', edge_attr)

        # print('distances, edge_attr EqBl: ', distances.shape, edge_attr.shape) # (33k, 1), (33k, 1)
        edge_attr = torch.cat([distances, edge_attr], dim=1) # concatenate edge features with sinusoidal embedding distances (..?)

        self.print_diagnostics('edge_attr_2', edge_attr)


        # update atom-type reprs. for L layers, get h(L) as output
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edge_index, edge_attr=edge_attr,
                                               node_mask=node_mask, edge_mask=edge_mask)

        # now we use the resulting h(L), and coordinate input, to compute the updated coordinates x(L)
        x = self._modules["gcl_equiv"](h, x, edge_index, coord_diff, coord_cross, edge_attr, # edge_attr = (.., 2)
                                       node_mask, edge_mask, update_coords_mask=update_coords_mask)

        # if node_mask is provided, then we make the resulting atom-type representations of said nodes 0

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        # return h(L) and x(L)
        return h, x


class EGNN(nn.Module):
    '''
    This module actually comes on top of the EquivariantBlock; it wraps the Block, and runs it multiple times as a whole for n_layers times;
    that means that the (2 times atom-type update module, 1 time coordinate update module) process/pipeline is run for n_layers amount of times,
    to get the ACTUAL h(L) and x(L), which should at this point be the vectors containing the exact noise that
    was added to the input x and h; the noise that, if we remove it, turns noised x and h into the
    slightly less noised h_(t-1) and x_(t-1); we compute the loss with the output of this model I believe.
    '''
    
    # inputs are the same as to the EquivariantBlock
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=3, attention=False,
                 norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, norm_constant=1, inv_sublayers=2,
                 sin_embedding=False, normalization_factor=100, aggregation_method='sum', reflection_equiv=True):
        super(EGNN, self).__init__()

        # 6 1 64 cuda SiLU() 4, out_node_nf=None, reflection_equiv=True, 'coordrangelayer' = 3.75
        print('egnn args: ', in_node_nf, in_edge_nf, hidden_nf, device, act_fn, n_layers, attention,
                 norm_diff, out_node_nf, tanh, coords_range, reflection_equiv, 'coordraneglayer', float(coords_range/n_layers))

        
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range/n_layers)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.reflection_equiv = reflection_equiv

        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = 2

        # print('edge feat nf: ', edge_feat_nf)

        # note: difference between E(3) impl. and DiffSBDD impl.; '+ in_edge_nf' was added --> what does it mean?
        # edge_feat_nf = edge_feat_nf + in_edge_nf # 3        


        # embedding layer for the atom-types (get initial embedding using fully conn. layer), of shape hidden_emb
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.zeros_(self.embedding.bias)
        
        # output fully conn. layer, mapping hidden dim. back to atom-type dimension size/shape
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        nn.init.xavier_uniform_(self.embedding_out.weight)
        nn.init.zeros_(self.embedding_out.bias)

        # add n_layers Equivariant blocks to module sequence
        for i in range(0, n_layers):
            self.add_module("e_block_%d" % i, EquivariantBlock(hidden_nf, edge_feat_nf=edge_feat_nf, device=device, # edge feat nf = 3
                                                               act_fn=act_fn, n_layers=inv_sublayers,
                                                               attention=attention, norm_diff=norm_diff, tanh=tanh,
                                                               coords_range=coords_range, norm_constant=norm_constant,
                                                               sin_embedding=self.sin_embedding,
                                                               normalization_factor=self.normalization_factor,
                                                               aggregation_method=self.aggregation_method,
                                                               reflection_equiv=self.reflection_equiv))
        self.to(self.device)
    
    def print_diagnostics(self, tens_name, inp_tens):
        with torch.no_grad():
            print(f"{tens_name} stats:")
            print("  norm:", inp_tens.norm().item())
            print("  min:", inp_tens.min().item())
            print("  max:", inp_tens.max().item())
            print("  mean:", inp_tens.mean().item())
            print("  std:", inp_tens.std().item())

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, update_coords_mask=None,
                batch_mask=None, edge_attr=None): # note: update_coords_mask, batch_mask AND edge_attr. added as part of DiffSBDD-EGNN
        # get h, x and adj. matrix+masks as inputs

        # Edit Emiel: Remove velocity as input
        # get rel. distances

        self.print_diagnostics('x', x)

        edge_feat, _ = coord2diff(x, edge_index) # note: this is called 'distances' in E(3) impl.
        # self.print_diagnostics('edge_feat', edge_feat)

        # print('edge index: ', edge_index, len(edge_index))

        # apply sinusoidal embedding if indicated
        if self.sin_embedding is not None:
            edge_feat = self.sin_embedding(edge_feat)
            # self.print_diagnostics('edge_feat_sin_embd', edge_feat)
        
        # concatenate edge feature reprs. to rel. distances, if they are provided; note: this concatenation does not happen in original E(3) impl.
        if edge_attr is not None:
            # self.print_diagnostics('egnn_edge_attr', edge_attr)
            
            # print('egnn sh: ', edge_feat.shape, edge_attr.shape)
            edge_feat = torch.cat([edge_feat, edge_attr], dim=1)

            # self.print_diagnostics('edge_feat_concat_attrs', edge_feat)

        # else:
        #     print('else egnn sh: ', edge_feat.shape, edge_attr) # feat, attr = (33856, 1), None
        
        # embed atom types into hidden dim.-sized continuous vector/tensor
        h = self.embedding(h)

        # self.print_diagnostics('h', h)

        # turn h(0) and x(0) into h(L) and x(L)
        for i in range(0, self.n_layers):
            h, x = self._modules["e_block_%d" % i](
                h, x, edge_index, node_mask=node_mask, edge_mask=edge_mask,
                edge_attr=edge_feat, update_coords_mask=update_coords_mask, # edge_attr/edge_feat = (33k, 1)
                batch_mask=batch_mask) # note: last two inputs added as part of DiffSBDD impl.

        # Important, the bias of the last linear might be non-zero
        # output embedding MLP ("decoder" module), to take hidden dim. of atom types back to atom-type dimension
        h_hidden = h.clone()

        h = self.embedding_out(h)

        # self.print_diagnostics('h_out', h)
        # self.print_diagnostics('h_hidden', h_hidden)

        # set resulting atom-types that we do not want to update to 0 (e.g. if the node/atom does not exist or we do not want to perform denoising/diffusion on them)
        if node_mask is not None:
            h = h * node_mask
            h_hidden = h_hidden * node_mask

        # return the resulting h(L) and x(L), which I think are now eps_h and eps_x, i.e. the predicted noise vectors for our input in a resulting timestep t->t-1
        return h, x, h_hidden

# helper class and functions
class SinusoidsEmbeddingNew(nn.Module):
    def __init__(self, max_res=15., min_res=15. / 2000., div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = 2 * math.pi * div_factor ** torch.arange(self.n_frequencies)/max_res
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()


def coord2diff(x, edge_index, norm_constant=1):
    '''
    Takes coordinates, and outputs the relative distances between every atom coordinate pair
    '''
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff/(norm + norm_constant)
    return radial, coord_diff


def coord2cross(x, edge_index, batch_mask, norm_constant=1):
    '''
    Performs a cross-product on the atom coordinates for the sake of the cross-MLP,
    to enable reflection sensitivity in the EGNN
    '''
    row, col = edge_index
    mean = x.mean(dim=0)
    # mean = unsorted_segment_sum(x, row, # batch_mask is row from edge_index in the other func.
    #                             num_segments=x.size(0),#batch_mask.max() + 1,
    #                             normalization_factor=None,
    #                             aggregation_method='mean')
    
    cross = torch.cross(x[row]-mean[batch_mask[row]],
                        x[col]-mean[batch_mask[col]], dim=1)
    norm = torch.linalg.norm(cross, dim=1, keepdim=True)
    cross = cross / (norm + norm_constant)
    return cross

def sum_except_batch(x):
    '''
    sum_except_batch aggregates the seq_len and emb dimensions, and sums over these two; one loss value per molecule is the result;
    i.e., aggregation over the representation's dimensions, as done to compute the (L2) loss, but leave the batch dim. intact.
    '''
    return x.view(x.size(0), -1).sum(-1)


def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
        
        -> takes a list of 'segment ids', which indicate which parts of the tensor have to be summed up together;
            if we have a tensor [0,1,2,3,4] and segment_ids = [0,0,1,1,1], we get [1, 9] as output. This can also
            be used on multi-dimensional tensors, in the following way:
            inp = [[1,2,3],[4,5,6],[7,8,9]], segment_ids = [0,0,1]
            -> output = [[5,7,9], [7,8,9]]
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result


def compute_drift_penalty(model, pretrained_weights, lambda_reg=0.001):
    '''
    Computes lambda regularization values for pretrained to finetuning predictions;
    model should be one of the encoders.
    '''

    pretrained_weights = {name: param.clone().detach()
                     for name, param in model.named_parameters()}


    penalty = 0
    for name, param in model.named_parameters():
        print(f'name: {name}, param val std: {param.std().item()}')
        if name in pretrained_weights:
            penalty += torch.norm(param - pretrained_weights[name])**2
            # print(f'penalty: {penalty}')

    return lambda_reg * penalty

# from mm_polymer, simply linear-> act -> linear to some desired out dim.
class NonLinearHead(nn.Module):
    """Head for simple classification tasks."""

    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
        hidden=None,
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.norm = nn.LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, hidden)
        # self.norm_2 = nn.LayerNorm(hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = activation_fn # utils.get_activation_fn(activation_fn)

        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

        

    def print_diagnostics(self, tens_name, inp_tens):
        with torch.no_grad():
            print(f"{tens_name} stats:")
            print("  norm:", inp_tens.norm().item())
            print("  min:", inp_tens.min().item())
            print("  max:", inp_tens.max().item())
            print("  mean:", inp_tens.mean().item())
            print("  std:", inp_tens.std().item())

    def forward(self, x):
        # self.print_diagnostics('x_not-normed', x)
        x = self.norm(x)
        # self.print_diagnostics('x_normed', x)
        x = self.linear1(x)
        x = self.activation_fn(x)
        # print('x inside nonlinear: ', x.shape)
        x = self.linear2(x)
        return x


### the language model
# this is just one line for now, chemberta with HF PretrainedModel

### wrapper for both models


# this will have a forward module that contains, on one side, a call to the (wrapper/dynamics functiona round the) EGNN module, and on another side,
# a call to the language model, and a call to a module that returns cross-aligned representations of the two. There will then be a class/'module', which calls this model's forward,
# takes the output representations, and computes the 3 loss values, summing these into one training signal/loss; this can then be passed to a (standard) training module;
# the loss functions will be BCE (EGNN), LM loss (LM), and contrastive loss (Cross-Alignment). The loss module will be different depending on the task;
# representation learning, property prediction, or diffusion, and we will need a property prediction head in the second case, and an entry into the E(3) diffusion
# model/wrapper, in the latter case. The first case we are working on is the first (reprs.), with the BCE etc. loss mentioned above.
class polyMRL(nn.Module):
    def __init__(self, in_node_nf, context_node_nf,
                 n_dims=3, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 condition_time=False, tanh=False, mode='egnn_dynamics', norm_constant=0,
                 inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum',
                 norm_values=None, norm_biases=None, include_charges=True, loss_type='l2',# taken from EnVariationalDiffusion() initialize function;
                 tokenizer=None, learned_noise=False, train_mode='property_prediction', freeze_encoders=True,
                 freeze_projections=False, pretrained_dir='output/polyMRL_qm9_test', cls_rep_3d='naive',
                 reflection_equiv=True, pretrain_checkpoint_dir_1d=None, pretrain_checkpoint_dir_3d=None,
                 use_layernorm=True, only_1d=False, only_3d=False, scaling_value=1.0
                 ):
        '''
        Initialization of model class; inputs are inputs and shapes for the two submodules (LM+EGNN), which
        need to be passed during training in args;

        in_node_nf: emb. dim. of node type representation h; in qm9, this is 6;
        context_node_nf: in case a property condition is passed, this is the emb. dim. of the property;
        n_dims: coordinate number of features; this is consistently 3 for us;
        hidden_nf: hidden dim. used by both GCL layer and EquivariantUpdate of the EGNN; defaults to 64;
        ..
        n_layers: how many EquivariantBlocks should be in our EGNN;
        ..
        '''
        super(polyMRL, self).__init__()

        print('model args: ', in_node_nf, context_node_nf, # 6, 0, 3, 64
                 n_dims, hidden_nf, device,
                 act_fn, n_layers, attention,
                 condition_time, tanh, mode, norm_constant,
                 inv_sublayers, sin_embedding, normalization_factor, aggregation_method,
                 norm_values, norm_biases, include_charges, loss_type,# taken from EnVariationalDiffusion() initialize function;
                 tokenizer, learned_noise)
        ## args
        self.in_node_nf = in_node_nf
        self.hidden_nf=hidden_nf
        self.n_dims = n_dims
        self.context_node_nf = context_node_nf
        self.device = device
        self.norm_values = norm_values
        self.norm_biases = norm_biases
        self.include_charges = include_charges
        self.loss_type = loss_type # if a different loss type is desired to be passed, we have to change the call to the loss function in the forward below!
        self.learned_noise = learned_noise
        self.train_mode = train_mode # pretrain, or property_prediction
        self.freeze_encoders = freeze_encoders
        self.freeze_projections = freeze_projections
        self.pretrained_dir = pretrained_dir
        self.cls_rep_3d = cls_rep_3d
        self.reflection_equiv = reflection_equiv
        self.pretrain_checkpoint_dir_1d = pretrain_checkpoint_dir_1d
        self.pretrain_checkpoint_dir_3d = pretrain_checkpoint_dir_3d
        self.use_layernorm = use_layernorm
        self.only_1d = only_1d
        self.only_3d = only_3d
        self.scaling_value=scaling_value
        
        # for target normalization
        self.partition='train'


        ### initialize LM language model, pretrained taken from the molgpt script
        # lm_config = RobertaConfig(vocab_size=tokenizer.vocab_size, block_size=tokenizer.model_max_length) # old (from MolGPT): 68 / tokenizer.model_max_length, -> we keep the block_size from pretraining as our finetuning block_size for now; optimally, we could dynamically select sequence lengths for the finetuning dataset, however this is not possible for our custom MolGPT setup
        # trying now:
        # config = RobertaConfig.from_pretrained("seyonec/ChemBERTa-zinc-base-v1") # wrong base, Roberta not used for this one
        config = AutoConfig.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        config.vocab_size=tokenizer.vocab_size # we cannot modify this with a pretrained model instance!
        config.block_size=tokenizer.model_max_length # we could modify this, but rather use model_max_length = default 512, with dynamic padding for now.
        # config.hidden_dropout_prob = 0.1  # example customization

        # not used for now
        # n_embd=args.n_embd,
        # n_layer= args.n_layer,
        # n_head= args.n_head,

        # trying now:
        # self.net_1d = RobertaForMaskedLM(lm_config)
        self.net_1d = AutoModelForMaskedLM.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")#, config=config)
        # self.net_1d = RobertaForMaskedLM.from_pretrained("seyonec/ChemBERTa-zinc-base-v1", config=config)


        ### initialize EGNN dynamics model
        self.net_dynamics = EGNN_dynamics_QM9( # = net_3d (can rename I think)
            in_node_nf=in_node_nf, context_node_nf=context_node_nf,
            n_dims=3, device=device, hidden_nf=hidden_nf,
            act_fn=torch.nn.SiLU(), n_layers=n_layers,
            attention=attention, tanh=tanh, mode=mode, norm_constant=norm_constant,
            inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
            normalization_factor=normalization_factor, aggregation_method=aggregation_method,
            reflection_equiv=self.reflection_equiv)

        # gamma network, used when setting forward noising is set to 'learned'
        # -> imported from en_diffusion.py/EnVariationalDiffusion script
        self.gamma = GammaNetwork()

        ### initialize module that compute cross-alignment representation
        # was 'seq_layer'
        self.contr_1d_layer = NonLinearHead( # intermediate (in attention-blocks) feedforward layer going from 768 (comes from where? -> I think from LM) to hidden 512 dim. (needed for loss computattion maybe..?)
            768, 512, act_fn
        )

        # was 'space_layer' # maybe add dropout or layer-norm to contr_3d_layer, for going from a tiny (6) dim. to high dim. (512 for example)
        self.contr_3d_layer = NonLinearHead( # another intermediate feedforward layer going from dim. 512 to 512
            self.in_node_nf, 512, act_fn # replaced 6 with self.in_node_nf
        )
        self.contr_3d_layer_hidden = NonLinearHead( # another intermediate feedforward layer going from dim. 512 to 512
            self.hidden_nf, 512, act_fn # replaced 6 with self.in_node_nf
        )

        self.contr_3d_layer_xh = NonLinearHead( # another intermediate feedforward layer going from dim. 512 to 512
            self.in_node_nf+self.context_node_nf+self.n_dims, 512, act_fn # self.context_node_nf can also be 0
        )


        # loading in 'pre-pretrained' encoders
        if self.train_mode=='pretrain' and self.pretrain_checkpoint_dir_1d is not None:
            # check if we have two checkpoint directories passed
            assert isinstance(self.pretrain_checkpoint_dir_1d, str)
            checkpoint_1d = torch.load(os.path.join(self.pretrain_checkpoint_dir_1d, "pretrained_encoders_proj_heads.pth"), map_location=self.device) # entire pretrained-model's state_dict
            self.net_1d.load_state_dict(checkpoint_1d["encoder_1d"], strict=False)
            print(f'loaded in 1d encoder weights, from: {self.pretrain_checkpoint_dir_1d}')

        if self.train_mode=='pretrain' and self.pretrain_checkpoint_dir_3d is not None:
            # check if we have two checkpoint directories passed
            assert isinstance(self.pretrain_checkpoint_dir_3d, str)
            checkpoint_3d = torch.load(os.path.join(self.pretrain_checkpoint_dir_3d, "pretrained_encoders_proj_heads.pth"), map_location=self.device) # entire pretrained-model's state_dict
            self.net_dynamics.load_state_dict(self.remove_mismatched_ckpt_layers(checkpoint_3d["encoder_3d"], self.net_dynamics.state_dict()), strict=False)
            print(f'loaded in 3d encoder weights, from: {self.pretrain_checkpoint_dir_3d}')


        self.class_head=None
        if self.train_mode=='property_prediction':
            if self.only_1d or self.only_3d:
                cls_dim_nf = 512
            else:
                cls_dim_nf=768
                # cls_dim_nf = self.in_node_nf
                # cls_dim_nf = 768+self.in_node_nf
                # cls_dim_nf = 1024

            # this projects from the concatenated CLS tokens down to one floating-point target
            self.layernorm_class = nn.LayerNorm(cls_dim_nf)

            # variant 1
            # self.dropout = nn.Dropout(0.1)
            # self.layer1 = nn.Linear(cls_dim_nf, cls_dim_nf)
            # self.act1 = nn.ReLU()
            # self.layer2 = nn.Linear(cls_dim_nf, 1)
            # self.class_head = nn.Sequential(self.dropout,
            #                                 self.layer1,
            #                                 self.act1,
            #                                 self.layer2
            #                                 )

            # variant 2
            # self.dropout = nn.Dropout(0.1)
            # self.layer1 = nn.Linear(cls_dim_nf, 2048)
            # self.act1 = nn.ReLU()
            # self.layer2 = nn.Linear(2048, 512)
            # self.act2 = nn.ReLU()
            # self.layer3 = nn.Linear(512, 1)
            # self.class_head = nn.Sequential(self.dropout,
            #                                 self.layer1,
            #                                 self.act1,
            #                                 self.layer2,
            #                                 self.act2,
            #                                 self.layer3
            #                                 )
            # variant 3
            self.dropout = nn.Dropout(0.1)
            self.layer1 = nn.Linear(cls_dim_nf, 1)
            self.class_head = nn.Sequential(self.dropout,
                                            self.layer1
                                            )

            # self.class_head = nn.Sequential(nn.Dropout(0.1),
            #                                 nn.Linear(cls_dim_nf, cls_dim_nf),
            #                                 nn.SiLU(),
            #                                 nn.Linear(cls_dim_nf, 1)
            #                                 )
            # self.class_head = nn.Sequential(self.dropout,
            #                                 self.layer1,
            #                                 self.act1,
            #                                 self.layer2
                                            # )
            self.pred_loss = torch.nn.HuberLoss() # MSELoss()

            # load in pretrained weights of the encoders and projection heads:
            # -> note: currently we ignore h-related weights because of mismatch; two options to improve this are 1) manual setting of pretrained h-weights (using e.g. weight[:,:,:6]); 2) creating one 'master-vocabulary' over all datasets, and training all datasets using this atom vocabulary directly;
            # checkpoint = torch.load(os.path.join(self.pretrained_dir, "pretrained_encoders_proj_heads.pth"), map_location=self.device) # entire pretrained-model's state_dict
            checkpoint = torch.load(os.path.join("outputs", "final_model_1d_PI1M-weightdecay-no_amp-fp32_epoch_4", "pretrained_encoders_proj_heads.pth"), map_location=self.device) # entire pretrained-model's state_dict
            # final_model_3d_PI1M-weightdecay-no_amp-fp32_epoch_5
            print('checkpoint model: ', checkpoint)

            self.net_1d.load_state_dict(checkpoint["encoder_1d"], strict=False) # pretrained and aligned 1d encoder if from combined pretrain model; if from 1d trained model only, this would be simply a 1d encoder (pretrained LM)
            self.gamma.load_state_dict(checkpoint['gamma_network'], strict=False)
            self.contr_1d_layer.load_state_dict(checkpoint["contr_1d_layer"], strict=False)

            # model components with atom-type repr. h that causes mismatch errors; remove pretrained embedding layers to allow pretrained weight loading
            self.net_dynamics.load_state_dict(self.remove_mismatched_ckpt_layers(checkpoint["encoder_3d"], self.net_dynamics.state_dict()), strict=False)
            self.contr_3d_layer.load_state_dict(self.remove_mismatched_ckpt_layers(checkpoint["contr_3d_layer"], self.contr_3d_layer.state_dict()), strict=False)
            self.contr_3d_layer_xh.load_state_dict(self.remove_mismatched_ckpt_layers(checkpoint["contr_3d_layer_xh"], self.contr_3d_layer_xh.state_dict()), strict=False)
            # NOTE: uncomment below line after pretraining with hidden included.
            # self.contr_3d_layer_hidden.load_state_dict(self.remove_mismatched_ckpt_layers(checkpoint["contr_3d_layer_hidden"], self.contr_3d_layer_hidden.state_dict()), strict=False)

            self.output_scale = nn.Parameter(torch.ones(1)*self.scaling_value)   # Init as 1.0*something
            self.output_shift = nn.Parameter(torch.zeros(1))  # Init as 0.0
            # >>> input = torch.randn(3, 5, requires_grad=True) -> watch the requires_grads!

            # if selected in args.freeze_encoders, freeze 1D and 3D encoders, and the CLS (/contrastive module) projection heads (everything but classification head)
            if self.freeze_encoders:
                self.freeze_encoders_func()

            if self.freeze_projections:
                self.freeze_projections_func()
            # self.r2_metric = R2Score()
            
            # TODO: include unfreeze_layers here!


        ### define language modelling loss function

        ### define L2/L1 denoising loss function

        ### define contrastive loss function

        ### write forward, where inputs smiles, 3d field are passed (other inputs..? check inputs to both e(3)/dynamics, and lm forward)
            ### in forward, compute adjacency matrix, perform Gaussian noising of the x and h inputs and concatenate these into xh;
            ### pass xh to egnn_dynamics, pass tokenized smiles to lm, run loss functions on the two corresponding output representations
            ### run contrastive alignment module and run contrastive loss function on aligned representations;
            ### combine three losses, and return the total loss L


        ###(combine 3 loss values into one loss (hyperparameter added to the input params) L
        ### return total loss value L)

    def freeze_encoders_func(self):
        # need to load the trained 1d and 3d encoders
        # NOTE: freezing behavior is a hyperparameter and can have several options, like not freezing, not freezing CLS projection heads, or only freezing early layers
        for param in self.net_1d.parameters():
            param.requires_grad = False
        for param in self.net_dynamics.parameters():
            param.requires_grad = False
        # in case we use learned epsilon, freeze gamma network.
        for param in self.gamma.parameters(): # in case the xh version is used;
            param.requires_grad = False

    def freeze_projections_func(self):
        # freezing CLS heads
        for param in self.contr_1d_layer.parameters():
            param.requires_grad = False
        for param in self.contr_3d_layer.parameters():
            param.requires_grad = False
        for param in self.contr_3d_layer_xh.parameters(): # in case the xh version is used;
            param.requires_grad = False
        for param in self.contr_3d_layer_hidden.parameters(): # in case the hidden h-repr. version is used;
            param.requires_grad = False
    
    def unfreeze_encoders_func(self):
        # need to load the trained 1d and 3d encoders
        # NOTE: freezing behavior is a hyperparameter and can have several options, like not freezing, not freezing CLS projection heads, or only freezing early layers
        for param in self.net_1d.parameters():
            param.requires_grad = True
        for param in self.net_dynamics.parameters():
            param.requires_grad = True
        # in case we use learned epsilon, freeze gamma network.
        for param in self.gamma.parameters(): # in case the xh version is used;
            param.requires_grad = True

    def unfreeze_projections_func(self):
        # freezing CLS heads
        for param in self.contr_1d_layer.parameters():
            param.requires_grad = True
        for param in self.contr_3d_layer.parameters():
            param.requires_grad = True
        for param in self.contr_3d_layer_xh.parameters(): # in case the xh version is used;
            param.requires_grad = True
        for param in self.contr_3d_layer_hidden.parameters(): # in case the hidden h-repr. version is used;
            param.requires_grad = True


    def print_diagnostics(self, tens_name, inp_tens):
        with torch.no_grad():
            print(f"{tens_name} stats:")
            print("  norm:", inp_tens.norm().item())
            print("  min:", inp_tens.min().item())
            print("  max:", inp_tens.max().item())
            print("  mean:", inp_tens.mean().item())
            print("  std:", inp_tens.std().item())

    def remove_mismatched_ckpt_layers(self, pretrained_state_dict, current_model_state_dict):
        '''
        Given a pretrained state dict of the model or component of the model,
        and the new model's (initial) state dict, return a new state dict, with
        only those weights left from the pretrained state dict that also occur in the new
        state dict, and that are identically-shaped to the new state dict's weights

        returns: new state dict, which can be loaded onto the new model's state dict,
                    without mismatch errors.
        '''
        new_state_dict = {}

        for k, v in pretrained_state_dict.items():
            if k in current_model_state_dict:
                if v.shape == current_model_state_dict[k].shape:
                    new_state_dict[k] = v
                else:
                    # Skip, shape mismatch
                    print(f"Skipping {k}: shape mismatch {v.shape} vs {current_model_state_dict[k].shape}")

        return new_state_dict

    def subspace_dimensionality(self, node_mask):
        """Compute the dimensionality on translation-invariant linear subspace where distributions on x are defined."""
        number_of_nodes = torch.sum(node_mask.squeeze(2), dim=1)
        return (number_of_nodes - 1) * self.n_dims

    def normalize(self, x, h, node_mask):
        '''
        Normalize the input positions x and node types h; applies node_mask as well
        '''
        x = x / self.norm_values[0] # DEFINE
        delta_log_px = -self.subspace_dimensionality(node_mask) * np.log(self.norm_values[0]) # DEFINE

        # Casting to float in case h still has long or int type.
        # if self.norm_values:
            # divide integer by this val.

        h_cat = (h['categorical'].float() - self.norm_biases[1]) / self.norm_values[1] * node_mask # DEFINE
        h_int = (h['integer'].float() - self.norm_biases[2]) / self.norm_values[2]

        if self.include_charges:
            h_int = h_int * node_mask

        # Create new h dictionary.
        h = {'categorical': h_cat, 'integer': h_int}

        return x, h, delta_log_px
    
    def normalize_2(self, x, h, node_mask):
        # compute 
        # print('1', x.shape,h['integer'].shape, node_mask.shape)
        node_mask = node_mask.float()
        node_mask_unsqueezed = node_mask#.unsqueeze(-1)  # (B, N, 1)
        # print('2', node_mask_unsqueezed.shape)
        # re-center -> done in remove_mean_with_mask (I believe).
        # mean = (x * node_mask_unsqueezed).sum(dim=1, keepdim=True) / node_mask_unsqueezed.sum(dim=1, keepdim=True).clamp(min=1)
        # x = x - mean

        # re-scale (->normalize)
        var = ((x**2) * node_mask_unsqueezed).sum(dim=1, keepdim=True) / node_mask_unsqueezed.sum(dim=1, keepdim=True).clamp(min=1)
        std = var.sqrt().clamp(min=1e-3)
        x = x / std

        # we dont normalize categorical for now..
        h_cat = (h['categorical'].float() / 4.) * node_mask # DEFINE -> added the 4

        # we are going to apply the node mask to h integers and normalize it by its global biggest value
        h_int_raw = h['integer'].float()  # shape (B, N)
        
        # node mask (no padding/CLS/asterisk tokens included) (only 'actual'/'physical' atoms)
        # atom_mask = (h_int_raw > 0).float() * node_mask  # (B, N) # old, for removing the -1/-2s
        h_int_norm = h_int_raw.clone()

        # re-scale by dividing by largest charge in dataset
        if self.norm_values:
            assert (isinstance(self.norm_values, int)) or (isinstance(self.norm_values, float))
            # h_int_norm = (h_int_raw / float(self.norm_values)) * atom_mask # old, zeros out CLS and * tokens (<0 charge (right..?))
            # we only normalize the positive (physical atom) charge values
            h_int_norm[h_int_norm != 0] = h_int_norm[h_int_raw != 0] / float(self.norm_values)
            h_int_norm = h_int_norm * node_mask

        return x, {'categorical': h_cat, 'integer': h_int_norm}


    # note: net_out is (our) 'eps_pred'
    def compute_error(self, net_out, eps):
        """
        Computes error, i.e. the most likely prediction of x.;
        function for L2 computation
        """
        # eps_t is concat of predicted coordinated noise eps (from x, 'vel' output from dynamics function;) and predicted node noise ('eps_h') (from h)
        # print('net out: ', net_out)
        # print('eps: ', eps)
        eps_t = net_out # shape (batch, seq_len, h+x) = (64, 25, 6+3)
        diffs = eps - eps_t


        # if we have virtual atoms, we mask them out so they do not get counted in the loss comp.:
        n_nodes_computed = eps_t.shape[1]
        if self.cls_rep_3d=='virtual_atom':
            bs, n_nodes, _ = eps_t.shape
            mask = torch.ones(bs, n_nodes, 1).to(self.device)
            mask[:,0,:] = 0.0
            
            diffs = diffs * mask # get virtual atom to 0 error value

            n_nodes_computed -= 1

        if self.loss_type == 'l2': # (old: if self.training and ); 'self.training' is an internal Torch attribute.
            denom = (self.n_dims + self.in_node_nf) * n_nodes_computed # (3 + e.g. 6) * seq_len/num_atoms -> why is that the reg. term we add to get the L2 loss?
            print('denom: ', denom)
            error = sum_except_batch(diffs ** 2) / denom
        else:
            error = sum_except_batch(diffs ** 2) / denom
        return error

        
    def contrastive_loss(self, X_1d, X_3d, temperature=0.07):
        """
        z1: (B, D) - LM embeddings
        z2: (B, D) - EGNN embeddings
        """
        B = X_1d.shape[0]

        # Cosine similarity matrix: [B, B]
        sim_matrix = torch.matmul(X_1d, X_3d.T) / temperature

        # Ground truth: diagonal elements are positives
        labels = torch.arange(B).to(X_1d.device)

        loss_1d_to_3d = F.cross_entropy(sim_matrix, labels)
        loss_3d_to_1d = F.cross_entropy(sim_matrix.T, labels)

        return (loss_1d_to_3d + loss_3d_to_1d) / 2

    def inflate_batch_array(self, array, target):
        """
        Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
        axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)

    # below 4 functions are for the adding of noise for forward 3d noising
    def sigma(self, gamma, target_tensor):
        """Computes sigma given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_tensor)

    def alpha(self, gamma, target_tensor):
        """Computes alpha given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_tensor)

    # these 2 fn's only extend the shape of sigma/alpha to the shape of the coordinate/target tensor (so we can add/multiply them to the target tensor)
    def sigma_fixed_schedule(self, sigma, target_tensor):
        """Computes output sigma tensor on a fixed sigma noise schedule; with fixed sigma value as input"""
        return self.inflate_batch_array(sigma, target_tensor)
    
    def alpha_fixed_schedule(self, sigma, target_tensor):
        """Computes output alpha tensor on a fixed sigma noise schedule; with fixed sigma value as input"""
        return self.inflate_batch_array(torch.sqrt(1-(sigma**2)), target_tensor)
    
    def sample_combined_position_feature_noise(self, n_samples, n_nodes, node_mask):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        z_x = sample_center_gravity_zero_gaussian_with_mask( # func. from helpers_from_edm/utils
            size=(n_samples, n_nodes, self.n_dims), device=node_mask.device,
            node_mask=node_mask)
        z_h = sample_gaussian_with_mask( # func. from helpers_from_edm/utils
            size=(n_samples, n_nodes, self.in_node_nf), device=node_mask.device,
            node_mask=node_mask)
        z = torch.cat([z_x, z_h], dim=2)
        return z


    ### start of the forward function
    # q: are we giving 'xh' or x and h separately to the forward function?
    # answer: x and h separately!
    def forward(self, combined_1d_3d): # smiles_tokens, smiles_attention_mask, x, h, node_mask, edge_mask, context):
        '''
        forward function of the combined model, taking in the string and 3D representations of a batch of molecules,
        computing 1D and 3D representatons, as well as the aligned representations,
        and computing losses for 1D, 3D, and contrastively-aligned representations.
        The losses are combined into one total loss value 'total_loss', which is returned (I think).
        
        inputs:
        x: input atom positions (potentially noise-augmented, and already mean-removed), (batch, seq_len, 3);
        h: input atom types, in dict form {'categorical': [[[0,0,1,0,0], ...]], 'integer': [[1,3,1,1,...]]},
            where one_hot indicates which atom it is, and integer indicates the atom number (periodic table nr) in int form;
        node_mask: binary mask indicating which nodes/atoms are not to be updated in the model, (batch, seq_len, 1); (copilot note: i.e. which atoms do not exist in the molecule?)
        edge_mask: binary mask indicating which edges are not to be updated in the model, (batch * seq_len^2, 1);
        context: property condition, (batch, seq_len, context_nf);
        target: (if applicable:) tensor (batch, seq_len, 1) in the setting where we perform property prediction (our downstream finetuning.

        smiles_tokens: tensor containing tokenized input ids for the molecules' SMILES sequences, (batch, max_seq_len);
        smiles_attention_mask: binary mask of (batch, max_seq_len) indicating which tokens in the input SMILES sequences are
            actual tokens, i.e. not padding, to indicate which tokens should be included in attention computation;

        

        --old:--
            xh: x and h concatenated, (batch, seq_len, 6+3) = (64, 25, 9);
                x here is the noised set of coordinates, of (batch, seq_len, 3),
                and h the atom types of embedded shape (64, 25, 6) or bigger than 6 if a conditioning property is added;
        '''
        batch_1d = combined_1d_3d['1d']
        batch_3d = combined_1d_3d['3d']

        logs = {}

        # print('batch_1d', batch_1d)
        # print('batch_3d', batch_3d)

        # print('devices: ')
        # for name, inp in batch_1d.items():
        #     if isinstance(inp, dict) and not (inp is None):
        #         for name2, inp2 in inp.items():
        #             print(name2, ': ', inp2.device)
        #     elif not (inp is None):
        #         print(name, ': ', inp.device)
        #     else:
        #         print('none: ', name, inp)
        # for name, inp in batch_3d.items():
        #     if isinstance(inp, dict) and not (inp is None):
        #         for name2, inp2 in inp.items():
        #             print(name2, ': ', inp2.device)
        #     elif not (inp is None):
        #         print(name, ': ', inp.device)
        #     else:
        #         print('none: ', name, inp)


        ## 1D
        # note: this is currently calling a Roberta-language model from HuggingFace, initialized empty and trained from scratch (not loaded with from_pretrained), which takes as inputs token ids plus an attention mask; how generalizable/extendable to other language models this is, I don't know
        seq_rep_output = self.net_1d(input_ids = batch_1d['input_ids'], attention_mask = batch_1d['attention_mask'], labels = batch_1d['labels'], output_attentions=True, output_hidden_states=True) # dictionary output containing {loss, logits, hidden_states, attentions}; .last_hidden_state attribute gives the final layer's hidden state, of shape (batch, max_seq_len, lm_hidden_nf)
        
        # seq_rep_output.loss, seq_rep_output.logits, seq_rep_output.hidden_states, seq_rep_output.attentions

        '''
        ## 3d
        # t, xh, node_mask, edge_mask, context
        x, h, node_mask, edge_mask, context = batch_3d['x'], batch_3d['h'], batch_3d['node_mask'], batch_3d['edge_mask'], batch_3d['context']

        batch_size, seq_len, coord_n_dims = x.size()

        edge_mask = edge_mask.view(batch_size, seq_len * seq_len)

        # run correctly masked assert function on the positions x, with the node mask
        assert_correctly_masked(x, node_mask) # from train_test()

        # normalize positions x and h, before concatenating the two into xh;
        x, h = self.normalize_2(x, h, node_mask) # , delta_log_px - from envariationaldiffusion()


        # add noising here with eps and sigma/alpha values
        # learned noise level using Gamma network (from DDPM method)
        if self.learned_noise:
            # compute gamma via the network, using a randomly initialized "timestep" t (taken from DDPM methodology)
            gamma = self.inflate_batch_array(self.gamma(np.random.randn()), x)

            # get sigma and alpha tensors in shape of x
            sigma = self.sigma(gamma, x).to(self.device) # added to(device) here
            alpha = self.alpha(gamma, x).to(self.device)

        # fixed noise level for sigma
        else:
            # scalar sigma_value has to be turned into a (1-dimensional) tensor first, then formed to match x's shape using the inflate_batch_array func.
            sigma_value = 0.05
            sigma_value = self.inflate_batch_array(torch.tensor([sigma_value], device=self.device), x) # note: check if the output of self.gamma(t) is actually a scalar!! (& not a tensor) -> if a tensor, make sigma_value a tensor manually and insert into inflate_batch_array()
            # print('sigma 1: ', sigma_value, sigma_value.shape) # tensor([[[0.05]]])
            
            # get sigma and alpha tensors in shape of x
            sigma = self.sigma_fixed_schedule(sigma_value, x)
            alpha = self.alpha_fixed_schedule(sigma_value, x)
            # print('sigma 2: ', sigma, sigma.shape) # tensor([[[0.05]]])
            # print('alpha 2: ', alpha, alpha.shape) # tensor([[[0.9987]]])


        # get standard-Gaussian noise in the shape of x
        eps = self.sample_combined_position_feature_noise(n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask)
        # eps_old = eps.clone().detach() # keep a copy of the noise var for checking the mask+noise below
        # print('eps: ', eps, eps.shape)

        # concatenate h['categorical'], h['integer'], and x into one tensor of dim. (3+len(vocab)+1), so that we have a combined tensor that we can noise (and pass to the EGNN) at once
        xh = torch.cat([x, h['categorical'], h['integer']], dim=2) # into (batch, seq_len, 9(=3+len(vocab)+1))
        # print('xh: ', xh, xh.shape)

        # if we use a virtual atom for contrastive + downstream learning, detach the virtual atom from gradient updates and set its noise to 0;
        if self.cls_rep_3d=='virtual_atom':
            xh_detached = xh.clone()
            xh_detached[:, 0, :] = xh_detached[:, 0, :].detach() # replace 0th atoms with detached 0th atoms (index 0 of seq_len axis)

            eps[:,0,:] = 0.0 # set added noise to 0 for virtual atom

        # print('xh shape 1:', xh, xh.shape) # (64, 25, 9) = (batch, seqlen, x+h)
        # add Gaussian noise to the combined 3D molecule representation xh
        xh = alpha * xh + sigma * eps
        # xh_old = alpha * xh + sigma * eps_old
        # print('xh shape 2:', xh, xh.shape) # (64, 25, 9) = (batch, seqlen, x+h)

        # masking check --> note: define self.n_dims (and the function)
        if self.cls_rep_3d != 'virtual_atom': # we dont run this check if we use virtual atom x_0 in the data, since we modify the noise added to xh then
            assert_mean_zero_with_mask(xh[:, :, :self.n_dims], node_mask)

        # print('xh sample (0:2): ', xh[0:1,:,:], 'total xh shape: ', xh.shape)

        # call 3D/EGNN model on correct inputs to get the predicted noise, eps, on both the positions x and atom types h
        # -> the output eps_pred is cat(vel, h_output), the output of EGNN_dynamics_QM9()'s forward
        print('xh shape: ', xh.shape)
        eps_pred, h_final, h_hidden = self.net_dynamics(None, xh, node_mask, edge_mask, context) # 'None' for 't'; net_out_3d is of the shape (batch, seq_len, 9) = xh_final
        

        ## computing the 1D and 3D CLS representations
        # X_a_L = h_final # final atom repr., of which we take atom 0 for the 3d representation (does that work....) (; shape = batch, num_atoms/seq_len, 6)
        X_a_L = h_final
        X_a_L = X_a_L * node_mask
        '''
        # 1D CLS & 3D CLS tokens


        # print('X_a_L: ', X_a_L)
        # print('X_a_L std: ', X_a_L.std().item()) # constantly around 0.7-1.1
        # print('X_a_L mean: ', X_a_L.mean().item())

        # print('seq_rep_output.hidden_states: ', seq_rep_output.hidden_states[-1])
        # print('seq_rep_output.hidden_states std: ', seq_rep_output.hidden_states[-1].std().item()) # constantly around 0.81
        # print('seq_rep_output.hidden_states: ', seq_rep_output.hidden_states[-1].mean().item())


        if self.cls_rep_3d=='virtual_atom':
            # x_3d alignment version 1: based on the 'empty' x_0 atom representations
            X_3d = X_a_L[:, 0, :] # the final atom representation, of virtual atom 0, for the entire batch, for a dimens. of (atom_vocab_size=)6; --> this is also (batch, hidden_dim (= batch, 6)) (I think 6, since coming from h_final)
            # X_3d = X_a_L.mean(dim=1) # = (batch, 6)
            # retrieve the last hidden state, of which we take the CLS token, and the entire batch and hidden dims.
            X_1d = seq_rep_output.hidden_states[-1][:, 0, :] # (Roberta) model's final CLS token repr. -> this becomes (batch, hidden_dim.)
        elif self.cls_rep_3d=='naive':
            # do not have to change X_a_L, because if virtual atom is not selected, it is automatically removed from the input batches' data
            X_3d = X_a_L[:, 0, :]
            # we do have to change the index in the lm hidden states from 0->1, representing the 1st physical token instead of the CLS token;
            X_1d = seq_rep_output.hidden_states[-1][:, 1, :] # (Roberta) model's final CLS token repr. -> this becomes (batch, hidden_dim.)
        elif self.cls_rep_3d in ['mean_pool', 'max_pool']:
            # take all atoms' reprs. besides the CLS token's to exclude that
            X_1d = seq_rep_output.hidden_states[-1][:, 1:, :]

            if self.cls_rep_3d == 'mean_pool':
                
                # X_3d = X_a_L.mean(dim=1)


                # print('X_3d: ', X_3d)
                X_1d = X_1d.mean(dim=1)
            elif self.cls_rep_3d == 'max_pool':
                X_3d = X_a_L.max(dim=1)
                # print('X_3d: ', X_3d)
                X_1d = X_1d.max(dim=1)

        # print('x1d: ', X_1d)
        print('x1d mean: ', X_1d.mean().item())
        print('x1d std: ', X_1d.std().item()) # constantly around 0.78

        # print('x3d: ', X_3d)
        # print('x3d mean: ', X_3d.mean().item())
        # print('x3d std: ', X_3d.std().item()) # 0.15 start of epoch and increases to 0.65 at the end of the epoch

        # print('x_1d output: ', seq_rep_output)#.hidden_states)
        # print(seq_rep_output.hidden_states.shape)
        # print('x_1d output shape [-1]: ', seq_rep_output.hidden_states[-1].shape)
        # print(seq_rep_output.hidden_states[-1].shape)
        
        # print('X_1d cls shape: ', X_1d.shape)

        # project 1D and 3D to (batch, hidden_dim.=e.g. 512)
        # self.print_diagnostics('X_3d', X_3d)
        self.print_diagnostics('X_1d', X_1d)

        target_norm = 1.0  # or: torch.sqrt(torch.tensor(x_1d.shape[-1])) for Xavier-like
        # X_1d = X_1d / X_1d.norm(dim=-1, keepdim=True) * target_norm
        # X_3d = X_3d / X_3d.norm(dim=-1, keepdim=True) * target_norm

        
        # NOTE: --> see what this (normalized) X_1d looks like (what is its std), and try if we can use this instead of X_1d_projected, as the CLS repr.
        # print('x1d normalized: ', X_1d)
        print('x1d normalized mean: ', X_1d.mean().item())
        print('x1d normalized std: ', X_1d.std().item()) # consistently 0.036

        # print('x3d normalized: ', X_3d)
        # print('x3d normalized mean: ', X_3d.mean().item())
        # print('x3d normalized std: ', X_3d.std().item()) # consistently around 0.22

        # X_1d = F.normalize(X_1d, dim=-1)
        # X_3d = F.normalize(X_3d, dim=-1)
        # self.print_diagnostics('X_3d_normed-scaled-1', X_3d)
        # self.print_diagnostics('X_1d_normed-scaled-1', X_1d)

        '''
        if X_a_L.shape[2]>50:
            print('using contr_3d_layer_hidden')
            X_3d_projected = self.contr_3d_layer_hidden(X_3d)

        else:
            X_3d_projected = self.contr_3d_layer(X_3d)

        X_1d_projected = self.contr_1d_layer(X_1d)
        
        
        print('X_1d_projected: ', X_1d_projected)
        print('X_1d_projected mean: ', X_1d_projected.mean().item())
        print('X_1d_projected std: ', X_1d_projected.std().item()) # around 1.8-3.5

        print('X_3d_projected: ', X_3d_projected)
        print('X_3d_projected mean: ', X_3d_projected.mean().item())
        print('X_3d_projected std: ', X_3d_projected.std().item())  # starts epoch around 0.04, then goes to like 0.09, then goes back to around 0.05 at the end of epochs (4)
        
        # self.print_diagnostics('X_3d_projected', X_3d_projected)
        # self.print_diagnostics('X_1d_projected', X_1d_projected)
        # print('X_1d_projected: ', X_1d_projected.shape)

        # normalize projected representations using their own mean - on advice of ChatGPT
        X_1d_projected = F.normalize(X_1d_projected, dim=-1)
        # X_3d_projected = F.normalize(X_3d_projected, dim=-1)
        
        print('X_1d_projected normalized: ', X_1d_projected)
        print('X_1d_projected normalized mean: ', X_1d_projected.mean().item())
        print('X_1d_projected normalized std: ', X_1d_projected.std().item()) # consistently at 0.044

        print('X_3d_projected normalized: ', X_3d_projected)
        print('X_3d_projected normalized mean: ', X_3d_projected.mean().item())
        print('X_3d_projected normalized std: ', X_3d_projected.std().item()) # consistently at 0.044


        self.print_diagnostics('X_3d_normed-2', X_3d_projected)
        self.print_diagnostics('X_1d_normed-2', X_1d_projected)
        # print('X_1d_projected normalized shape: ', X_1d_projected.shape)
        '''
        ## loss computations
        if self.train_mode=='pretrain':
            # compute 3D i.e. denoising (l2) error -> if we use virtual atoms, the 0-th atoms are excluded from this denoising noise computation
            loss_3d_batch = self.compute_error(eps_pred, eps) # l2 loss
            # print('3d losses all: ', loss_3d_batch)
            loss_3d = loss_3d_batch.mean(0)

            # get 1D loss from LM output object
            loss_1d = seq_rep_output.loss # note: turn on output-hidden_states!!

            ## cross-alignment
            ## NOTE: there is no atom "x0" in the moleculesof qm9, so x3d_projected is bound to not contain much useful information on the entire molecule;
            ## try different approaches, like mean-pooling/aggregation of the embedding, perhaps? Or a different method?
            ## TODO: cross-alignment version 2; based on x+h mean-pooling
            loss_contrastive = self.contrastive_loss(X_1d_projected, X_3d_projected)

            total_loss = 0.3*loss_1d + 0.45*loss_3d + 0.2*loss_contrastive

            # print and log individual+combined losses
            print('loss_3d : ', loss_3d.item())
            print('loss_1d: ', loss_1d.item())
            print('loss_contrastive: ', loss_contrastive.item())
            print('total loss: ', total_loss.item())
            logs['loss_1d']=loss_1d.item()
            logs['loss_3d']=loss_3d.item()
            logs['loss_contrastive']=loss_contrastive.item()
            logs['total_loss']=total_loss.item()


        # TODO: freeze encoders (or give option to) when doing prop. pred.
        # doing property prediction, compute MSE as loss
        elif self.class_head:
            targets = batch_3d['target'] # target is of shape (batch, seq_len,1 )
            # we need mad and mean of train dataloader for target normal.
            mad = batch_3d['mad']
            mean = batch_3d['mean']

            # create (concatenated) cls representation
            if self.only_1d: # only 1d encoder's cls
                print('just 1d')
                cls_representation = X_1d_projected
            elif self.only_3d: # only 3d encoder's cls
                print('just 3d')
                cls_representation = X_3d_projected
            else: # combining both modalities
                print('combined 1d-3d')
                cls_representation = X_1d#torch.cat([X_1d, X_3d], dim=-1)
                # cls_representation = torch.cat([X_1d_projected, X_3d_projected], dim=-1)

                cls_representation_testing = self.dropout(cls_representation)
                print(f"Before dropout: std: {(cls_representation).std():.4f}, mean: {(cls_representation).mean():.4f}, norm: {(cls_representation).norm():.4f}")
                print(f"After dropout: std: {cls_representation_testing.std():.4f}, mean: {cls_representation_testing.mean():.4f}, norm: {cls_representation_testing.norm():.4f}")
                cls_representation_testing = self.layer1(cls_representation_testing)
                print(f"After linear 1: std: {cls_representation_testing.std():.4f}, mean: {cls_representation_testing.mean():.4f}, norm: {cls_representation_testing.norm():.4f}")
                # cls_representation_testing = self.act1(cls_representation_testing)
                # print(f"After activation: std: {cls_representation_testing.std():.4f}, mean: {cls_representation_testing.mean():.4f}, norm: {cls_representation_testing.norm():.4f}")
                # cls_representation_testing = self.layer2(cls_representation_testing)
                # print(f"After linear 2: std: {cls_representation_testing.std():.4f}, mean: {cls_representation_testing.mean():.4f}, norm: {cls_representation_testing.norm():.4f}")
                # cls_representation_testing = self.act2(cls_representation_testing)
                # print(f"After activation 2: std: {cls_representation_testing.std():.4f}, mean: {cls_representation_testing.mean():.4f}, norm: {cls_representation_testing.norm():.4f}")
                # cls_representation_testing = self.layer3(cls_representation_testing)
                # print(f"After linear 3: std: {cls_representation_testing.std():.4f}, mean: {cls_representation_testing.mean():.4f}, norm: {cls_representation_testing.norm():.4f}")

            with torch.no_grad():
                print(f"Before dropout - 5.0: std: {(cls_representation*5.0).std():.4f}, mean: {(cls_representation*5.0).mean():.4f}, norm: {(cls_representation*5.0).norm():.4f}")
                cls_representation_testing = self.dropout(5.0*cls_representation)
                print(f"After dropout - 5.0: std: {cls_representation_testing.std():.4f}, mean: {cls_representation_testing.mean():.4f}, norm: {cls_representation_testing.norm():.4f}")
                cls_representation_testing = self.layer1(cls_representation_testing)
                print(f"After linear 1 - 5.0: std: {cls_representation_testing.std():.4f}, mean: {cls_representation_testing.mean():.4f}, norm: {cls_representation_testing.norm():.4f}")
                # cls_representation_testing = self.act1(cls_representation_testing)
                # print(f"After activation - 5.0: std: {cls_representation_testing.std():.4f}, mean: {cls_representation_testing.mean():.4f}, norm: {cls_representation_testing.norm():.4f}")
                # cls_representation_testing = self.layer2(cls_representation_testing)
                # print(f"After linear 2 - 5.0: std: {cls_representation_testing.std():.4f}, mean: {cls_representation_testing.mean():.4f}, norm: {cls_representation_testing.norm():.4f}")
                # cls_representation_testing = self.act2(cls_representation_testing)
                # print(f"After activation 2 - 5.0: std: {cls_representation_testing.std():.4f}, mean: {cls_representation_testing.mean():.4f}, norm: {cls_representation_testing.norm():.4f}")
                # cls_representation_testing = self.layer3(cls_representation_testing)
                # print(f"After linear 3 - 5.0: std: {cls_representation_testing.std():.4f}, mean: {cls_representation_testing.mean():.4f}, norm: {cls_representation_testing.norm():.4f}")

            # consistently around 0.04
            # print('cls repr. :', cls_representation.std(), cls_representation.mean(), cls_representation.norm(), cls_representation)
            cls_representation =5.0 *  cls_representation # 
            print('cls repr. :', cls_representation.std(), cls_representation.mean(), cls_representation.norm())#, cls_representation)

            if self.use_layernorm:
                # cls_representation = self.layernorm_class(cls_representation) # should bring cls to mean 0, std 1 (instead of like std=0.05 that comes out of (projected) encoded reprs.)
                # consistently around 0.994 - 0.997
                print('not using layernorm for now')
                # print('cls repr. 2:', cls_representation.std(), cls_representation.mean(), cls_representation)

            # # correlation test:

            # correlations = []
            # x_1d_flat = cls_representation.view(cls_representation.size(0), -1)  # Flatten if needed
            # for dim in range(min(50, x_1d_flat.size(1))):  # Test first 50 dims
            #     corr = torch.corrcoef(torch.stack([x_1d_flat[:, dim],  ((targets-mean)/mad).squeeze()]))[0, 1]
            #     if not torch.isnan(corr):
            #         correlations.append(abs(corr.item()))

            # max_corr = max(correlations) if correlations else 0
            # print(f"max correlation between x_1d and normalized targets: {max_corr:.4f}")
            # print('correlations: ', correlations)



            # print('cls repr shape: ', cls_representation.shape)
            # cls_representation = torch.randn_like(cls_representation)
            # print(f'randomly initialized cls representation: std: {cls_representation.std().item()}, norm: {cls_representation.norm().item()}, start: {cls_representation[3,:]}')
            # cls_representation_constant = torch.ones_like(cls_representation)
            # print(f'randomly initialized cls representation: std: {cls_representation_constant.std().item()}, norm: {cls_representation_constant.norm().item()}, start: {cls_representation_constant[3,:]}')

            # torch.corrcoef(cls_representation, cls_representation_constant)

            

            prediction = self.class_head(cls_representation) # prediction is now also of shape (batch, seq_len, 1)
            # starts around 0.5, decreases fast towards like 0.0040 at the minimum
            # print('predictions 1:', prediction.std(), prediction.mean(), prediction)
            # add learned scaling parameter
            # prediction = self.output_scale * prediction + self.output_shift
            # print('output_scale: ', self.output_scale)
            # print('output_shift: ', self.output_shift)


            print('current partition:' , self.partition)
            if self.partition=='train':
                print('training partition')
                print('targets: ', targets.std(), targets.mean(), targets[:5])
                print('mean: ', mean)
                print('mad: ', mad)
                print('targets-mean/mad: ', ((targets-mean)/mad).std(),((targets-mean)/mad).mean(), ((targets-mean)/mad)[:5])
                print('prediction: ', prediction.std(), prediction.mean(), prediction[:5])
                print('predictions if they were scaled to target dist.: ', (mad*prediction+mean).std(), (mad*prediction+mean).mean(), (mad*prediction+mean)[:5])

                print('prediction.shape: ', prediction.shape)
                print('targets.shape: ', targets.shape)

                if len(prediction.shape)>1:
                    print('squeezing prediction tensor')
                    prediction=prediction.squeeze()
                    print('prediction squeezed: ', prediction.std(), prediction.mean(), prediction[:5])
                
                if len(targets.shape)>1:
                    print('squeezing targets tensor')
                    targets=targets.squeeze()
                    print('targets squeezed: ', targets.std(), targets.mean(), targets[:5])
                    

                assert len(prediction.shape)==1
                assert len(targets.shape)==1
                # try to include this penalty term
                # drift_penalty = compute_drift_penalty(self.net_1d)
                # print('drift_penalty: ', drift_penalty)
                # mse loss (optionally plus penalty term)

                # note: this is, in the current case, NOT the (R)MSE loss! But e.g. a Huber loss value. So we need to log and evaluate on the RMSE values still.
                total_loss = self.pred_loss(prediction, (targets-mean)/mad)# + drift_penalty
                # total_loss = self.pred_loss(prediction, targets)# + drift_penalty

                print('total loss (Huber): ',total_loss)
                
                rmse_n = root_mean_squared_error(((targets-mean)/mad).cpu().detach().numpy(), prediction.cpu().detach().numpy())
                r2_n = r2_score(((targets-mean)/mad).cpu().detach().numpy(), prediction.cpu().detach().numpy())
                print("batch's RMSE loss and R2 on normal. dist.: ",  rmse_n, r2_n)
                rmse_un = root_mean_squared_error(targets.cpu().detach().numpy(), (mad*prediction+mean).cpu().detach().numpy())
                r2_un = r2_score(targets.cpu().detach().numpy(), (mad*prediction+mean).cpu().detach().numpy())
                print("batch's RMSE loss and R2 on unnormal./target dist.: ",  rmse_un, r2_un)

                # self.r2_metric.update(prediction, (targets-mean)/mad)
                # r2 = metric.compute().item()

            elif self.partition in ['valid', 'test']:
                print('valid/test partition')
                print('original prediction: ', prediction.std(), prediction.mean(),prediction[:5])
                print('mean: ', mean)
                print('mad: ', mad)
                print('scaled prediction (mad*prediction+mean): ',(mad*prediction+mean).std(),(mad*prediction+mean).mean(), (mad*prediction+mean)[:5])
                print('targets: ',targets.std(), targets.mean(), targets[:5])
                print('targets if they were scaled to training dist.: ', ((targets-mean)/mad).std(), ((targets-mean)/mad).mean(), ((targets-mean)/mad)[:5])



                if len(prediction.shape)>1:
                    print('squeezing prediction tensor')
                    prediction=prediction.squeeze()
                    print('original prediction squeezed: ', prediction.std(), prediction.mean(),prediction[:5])

                if len(targets.shape)>1:
                    print('squeezing targets tensor')
                    targets=targets.squeeze()
                    print('targets squeezed: ',targets.std(), targets.mean(), targets[:5])

                assert len(prediction.shape)==1
                assert len(targets.shape)==1

                total_loss = self.pred_loss(mad*prediction+mean, targets)
                # total_loss = self.pred_loss(prediction, targets)
                print('total loss: ', total_loss)
                
                rmse_n = root_mean_squared_error(((targets-mean)/mad).cpu().detach().numpy(), prediction.cpu().detach().numpy())
                r2_n = r2_score(((targets-mean)/mad).cpu().detach().numpy(), prediction.cpu().detach().numpy())
                print("batch's RMSE loss and R2 on normal. dist.: ",  rmse_n, r2_n)
                rmse_un = root_mean_squared_error(targets.cpu().detach().numpy(), (mad*prediction+mean).cpu().detach().numpy())
                r2_un = r2_score(targets.cpu().detach().numpy(), (mad*prediction+mean).cpu().detach().numpy())
                print("batch's RMSE loss and R2 on unnormal./target dist.: ",  rmse_un, r2_un)

                
                print('current predictions vs targets:')
                for (valu,targ) in zip(mad*prediction+mean, targets):
                    print(valu.item(), targ.item(), '---> difference: ', abs(valu.item()-targ.item()))

            # >>> loss.backward()
            logs['total_loss'] = total_loss
            logs['property_predictions'] = prediction.std().item()
            logs['rmses'] = rmse_un
            logs['r2s'] = r2_un

        # print('intermediate test for param values: ')
        # for (n,param) in self.net_1d.named_parameters():
        #     print(f'parameter name h: {n}, parameter std: {param.std().item()}, parameter value: {param}')#, std:', param.std()) if param.grad is not None else (f'parameter {n} has no grad'))
        return total_loss, logs, self.net_1d.named_parameters()



        # check 3d hidden_nf (--> = 9), make sure 3d and 1d end up in same hidden dim for alignment computation; -> shouldnt we just compute 1d and 3d loss on their default hidden sizes, and then project to the same dimension for alignment?


        # for losses, take output as-is of 3D, and of 1D shape, and compute losses on them; compute the aligned representation using projected representations on a shared space, e.g. 512 big; contrastive loss then on aligned representations


        # compute 3D L2 loss value using 'net_out_3d' and 'eps'



    
    # ### warning: we renamed '_forward' to 'forward'!
    # def forward():


        
    #     # Compute gamma_s and gamma_t via the network.

    #     sigma = self.inflate_batch_array(self.gamma(t), x)

    #     # Compute alpha_t and sigma_t from gamma.
    #     ### note: we can use the alpha and sigma functions to compute the needed alpha and sigma for our constant noise computation,
    #     #  just need to see to use the sigmoid in there or not; without sigmoid, we take sqrt(0.05) for a 'sigma' value of 0.05
    #     #  which respects the equation sigma^2+alpha^2 = 1; with sigmoid, we transform the 0.05 'sigma' value into a value between [0,1],
    #     #  which for 0.05 means 0.5125 and 0.4875; the square root is taken on these again to get
    #     #  the corresponding alpha and sigma to transform the input xh with; maybe test out both with and without the sigmoid
    #     alpha_t = self.alpha(gamma_t, x)
    #     sigma_t = self.sigma(gamma_t, x)

    #     # note: I believe here we add the Gaussian noise, computed in forward noising, to the input sample x (=coordinates); noise is in variable epsilon 'eps', which we add using alpha and sigma to the concatenated 'xh'
        
    #     # Sample zt ~ Normal(alpha_t x, sigma_t)
    #     eps = self.sample_combined_position_feature_noise(
    #         n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask)
    #     print('eps: ', eps, eps.shape if isinstance(eps, torch.Tensor) else 'eps is not a tensor')

    #     # note: we actually concatenate the node type information (.. and charges..?) with the atom coordinates x, which means we noise these as well; does that mean we add noise to a one-hot encoded vector..?

    #     # Concatenate x, h[integer] and h[categorical].
    #     xh = torch.cat([x, h['categorical'], h['integer']], dim=2) # (64, 25, 9)

    #     # note: the noised data/coordinate+node type information is called 'z_t', i.e. a representation of the input at some (sampled) timestep t

    #     # Sample z_t given x, h for timestep t, from q(z_t | x, h)
    #     z_t = alpha_t * xh + sigma_t * eps





##### scripts in edm(3):
### train_test:
# remove_mean_with_mask(x)  --> new train loop (CHECK)
# augmentation with noise (x)  --> new train loop (CHECK)
# augmentation with rotation (x)  --> new train loop (CHECK)
# prepare and add context ('context')  --> new train loop (CHECK)

### compute_loss_and_nll:
# edge_mask.view(batch * seq_len^2)  --> new forward (CHECK)

### EnVariationalDiffusion:
# (forward) normalize (x, h, node_mask)  --> new forward
# (compute_loss) compute noise level; sample_combined_position_feature_noise(input-sizes)  --> new forward
# (compute_loss) concat(x, h)  --> new forward
# (compute_loss) add eps (Gaussian noise) to xh --> new forward


##### LM:

### (qm9) tokenizing is done in dataset_class
### (qm9) tokenized batch, so input_ids with attention_mask, is returned by dataloader-iter.;
### 


### other todos:
    ### write the (simplistic) trainer function;
    ### combine the dataloaders of QM9 and mol_gpt
    ### add the reflection equivariance to the EGNN module as an option/additive sum
    ### make sure the LM takes the same property condition appended to the molecule string as the EGNN model is taking;
    
    
    
    
# class EGNN_dynamics_QM9(nn.Module):
#     def __init__(self, in_node_nf, context_node_nf,
#                  n_dims, hidden_nf=64, device='cpu',
#                  act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
#                  condition_time=True, tanh=False, mode='egnn_dynamics', norm_constant=0,
#                  inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum'):
#         super().__init__()

#         print("EGNN dynamics: in_node_nf, context_node_nf,n_dims, hidden_nf=64, device=cpu, act_fn=torch.nn.SiLU(), n_layers=4, attention=False,condition_time=True")
#         print(in_node_nf, context_node_nf,
#                  n_dims, hidden_nf, device,
#                  act_fn, n_layers, attention,
#                  condition_time)
        

# copilot suggestion for polyMRL init:
'''
def __init__(self, egnn_model, lm_model, cross_alignment_model):
        super(polyMRL, self).__init__()
        self.egnn_model = egnn_model
        self.lm_model = lm_model
        self.cross_alignment_model = cross_alignment_model

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None,
                update_coords_mask=None, batch_mask=None, edge_attr=None):
        
        
        # call the EGNN model
        egnn_output = self.egnn_model(h, x, edge_index, node_mask=node_mask,
                                      edge_mask=edge_mask,
                                      update_coords_mask=update_coords_mask,
                                      batch_mask=batch_mask,
                                      edge_attr=edge_attr)
        
        # call the language model
        lm_output = self.lm_model(h)

        # call the cross-alignment model
        cross_output = self.cross_alignment_model(egnn_output, lm_output)

        return egnn_output, lm_output, cross_output
'''