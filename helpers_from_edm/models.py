import torch
import torch.nn as nn
from model.egnn_new import EGNN, GNN
from helpers_from_edm.utils import remove_mean, remove_mean_with_mask
import numpy as np


class EGNN_dynamics_QM9(nn.Module):
    def __init__(self, in_node_nf, context_node_nf,
                 n_dims, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 condition_time=True, tanh=False, mode='egnn_dynamics', norm_constant=0,
                 inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum'):
        super().__init__()

        print("EGNN dynamics: in_node_nf, context_node_nf,n_dims, hidden_nf=64, device=cpu, act_fn=torch.nn.SiLU(), n_layers=4, attention=False,condition_time=True")
        print(in_node_nf, context_node_nf,
                 n_dims, hidden_nf, device,
                 act_fn, n_layers, attention,
                 condition_time)

        self.mode = mode
        if mode == 'egnn_dynamics':
            self.egnn = EGNN(
                in_node_nf=in_node_nf + context_node_nf, in_edge_nf=1,
                hidden_nf=hidden_nf, device=device, act_fn=act_fn,
                n_layers=n_layers, attention=attention, tanh=tanh, norm_constant=norm_constant,
                inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method)
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

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)
        return fwd

    def unwrap_forward(self):
        return self._forward

    def _forward(self, t, xh, node_mask, edge_mask, context):
        '''
        t: timestep, scalar
        xh: x and h concatenated, (batch, seq_len, 6+3) = (64, 25, 9);
        x here is the noised set of coordinates, of (batch, seq_len, 3)
        
        '''
        bs, n_nodes, dims = xh.shape

        h_dims = dims - self.n_dims
        print('h_dims: ', h_dims) # 6, = len of vocab, i.e. nr. of possible atoms in dataset

        # build (fully-connected?) adjacency matrix
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [x.to(self.device) for x in edges] # (40000, ), (40000, )

        print('edges: ', edges)
        print('edges 0: ', edges[0].shape)

        node_mask = node_mask.view(bs*n_nodes, 1)
        edge_mask = edge_mask.view(bs*n_nodes*n_nodes, 1)
        print('nodemask edgemask reordered: ', node_mask.shape, edge_mask.shape)
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

        # concat timestep t to h in case we condition on timestep t
        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t.view(bs, 1).repeat(1, n_nodes)
                h_time = h_time.view(bs * n_nodes, 1)
            print('h time: ', h_time)
            h = torch.cat([h, h_time], dim=1)
            print('h plus h time: ', h, h.shape) # (1600, 7)

        # if we use property condition, concat this to h also on last dim., so we get h=(batch*seq_len, vocab_size+1(for time) + cond_num_feat)
        if context is not None:
            # We're conditioning, awesome!
            context = context.view(bs*n_nodes, self.context_node_nf)
            print('context: ', context)
            h = torch.cat([h, context], dim=1)
            print('h plus context: ', h.shape)

        # run the actual EGNN on the input node types/featurs, coordinates, edge lists, node and edge masks
        if self.mode == 'egnn_dynamics':
            h_final, x_final = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask)
            print('h final: ', h_final, h_final.shape)
            print('x final: ', x_final, x_final.shape)
            
            # the velocity ('momentum'?) is the final coordinate representation minus the initial coordinate representation (of the not-masked atoms)
            # note: x_final is clean predicted coordinate, x is input noisy coordinate;
            # the difference is the predicted coordinate noise 'eps'
            vel = (x_final - x) * node_mask  # This masking operation is redundant but just in case

        elif self.mode == 'gnn_dynamics':
            xh = torch.cat([x, h], dim=1)
            output = self.gnn(xh, edges, node_mask=node_mask)
            vel = output[:, 0:3] * node_mask
            h_final = output[:, 3:]

        else:
            raise Exception("Wrong mode %s" % self.mode)

        if context is not None:
            # Slice off context size:
            h_final = h_final[:, :-self.context_node_nf]

        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :-1]

        vel = vel.view(bs, n_nodes, -1)

        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan, resetting EGNN output to zero.')
            vel = torch.zeros_like(vel)

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
            # back from (batch*seq_len, h_num_feat) to (batch, seq_len, h_num_feat)
            h_final = h_final.view(bs, n_nodes, -1) # = (64, 25, 6); not sure why the time dimension (+1) is not in here; copilot says it is not needed for the diffusion model?
            
            return torch.cat([vel, h_final], dim=2)

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
