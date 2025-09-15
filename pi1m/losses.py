import torch


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(dim=-1)


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask)).abs().sum().item() < 1e-8


def compute_loss_and_nll(args, generative_model, nodes_dist, x, h, node_mask, edge_mask, context):
    bs, n_nodes, n_dims = x.size()

    print('bs, n_nodes, n_dims', bs, n_nodes, n_dims)

    # in the qm9 setting, we work with a EGNN, EGNN_QM9dynamics function and EnVariationalDiffusion model; args.probabilistic_model is set to 'diffusion'
    # ; does this mean that in this function, we get the forward-noised samples of the data, and not the un-noised variants of that data instead?
    # how could we test whether this is the case...?
    if args.probabilistic_model == 'diffusion':
        edge_mask = edge_mask.view(bs, n_nodes * n_nodes)
        print('edge mask reordered ', edge_mask.shape)

        assert_correctly_masked(x, node_mask)

        # Here x is a position tensor, and h is a dictionary with keys
        # 'categorical' and 'integer'.
        # note: this is the EGNN Dynamics function (I believe)
        # note: log here and print x, h, node_mask, edge_mask and context shapes
        # actually, I think this is the full EnDiffusion model that we push the inputs through here;
        # and the single output is the corresponding loss value we train on (in the train_test function loop).
        nll = generative_model(x, h, node_mask, edge_mask, context)
        print('nll: ', nll)

        # mean nr. of atoms per molecule (I think)
        N = node_mask.squeeze(2).sum(1).long()

        log_pN = nodes_dist.log_prob(N)

        assert nll.size() == log_pN.size()
        print(log_pN)
        # note: not sure if we need log_p(N) in the loss calculation right now -it seems to be a term related to the distribution of node information of the input batch - but
        # keep it in mind since it could be that we should include it in the model's loss calculation (maybe it makes the loss better)
        nll = nll - log_pN

        # Average over batch.
        print('nll shape: ', nll.shape)
        nll = nll.mean(0)
        print('nll shape 2: ', nll.shape)

        reg_term = torch.tensor([0.]).to(nll.device)
        mean_abs_z = 0.
    else:
        raise ValueError(args.probabilistic_model)

    return nll, reg_term, mean_abs_z
