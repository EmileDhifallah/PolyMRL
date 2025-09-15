import torch
import numpy as np


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        # think abouot adding torch.no_grad here
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def sum_except_batch(x):
    return x.reshape(x.size(0), -1).sum(dim=-1)


def remove_mean(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    x = x - mean
    return x


def remove_mean_with_mask(x, node_mask):
    '''
    Normalize molecule coordinates by mean-subtraction (translation norm.), centering atom
    coordinates around center/origin (0,0,0), while taking into account node mask,
    and also applying node mask at the end.
    '''
    # get masked positions' tokens (should be 0) 
    # print('1-node_mask',1-node_mask)
    # print('x*that.abs',(x * (1 - node_mask)).abs())
    # print('sumthat',(x * (1 - node_mask)).abs().sum().item())
    
    masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item() # should be 0. ..?
    # check whether masked tokens are indeed (approx.) 0
    # assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'

    # getting the number of actual atoms per molecule; N is the nr. of 'active' nodes in a molecule
    N = node_mask.sum(1, keepdims=True)

    # now we sum over dim 1 = sequence dimension of the molecule; this means that we add up all the coordinates of a molecule for both the x, y and z dimensions, and get essentially the 'middle-most'/summed coordinates of all atoms in the molecule (set of 3 values per molecule);
    # this gives us the average positions of a molecule! set of 3 coordinates per molecule that are the averages of the molecule's atom positions
    mean = torch.sum(x, dim=1, keepdim=True) / N
    # subtract the mean coordinates from every set of atom coordinates of the molecule; this centers the coordinates aroudn the origin (0,0,0)
    x = x - mean * node_mask

    # return normalized/centered atom positions x
    return x


def assert_mean_zero(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    assert mean.abs().max().item() < 1e-4


def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
    assert_correctly_masked(x, node_mask)
    largest_value = x.abs().max().item()
    error = torch.sum(x, dim=1, keepdim=True).abs().max().item()
    rel_error = error / (largest_value + eps)
    assert rel_error < 1e-2, f'Mean is not zero, relative_error {rel_error}'


def assert_correctly_masked(variable, node_mask):
    '''
    Checks whether all values that are masking tokens according to the mask,
    are in also actually masked, and that their value is now (within an epsilon of error) 0.
    '''
    # print('variable: ', variable)
    # print('node_mask: ', node_mask)
    # print('check val: ', (variable * (1 - node_mask)).abs().max().item())

    # print('1-node_mask',1-node_mask)
    # print('x*that.abs',(variable * (1 - node_mask)).abs())
    # print('max of that',(variable * (1 - node_mask)).abs().max().item())
    
    # if not (variable * (1 - node_mask)).abs().max().item() < 1e-4:
    #     print("assert failed hard, molecule showing:")
    #     for i,mask in enumerate(node_mask):
    #         print(f'mask: {i}', mask)
    #     for i,ch in enumerate(variable):
    #         for j,coord in enumerate(ch):
    #             print(f'coord {j} in mol {i}', coord)
  

    # assert (variable * (1 - node_mask)).abs().max().item() < 1e-4, \
        # 'Variables not masked properly.'


def center_gravity_zero_gaussian_log_likelihood(x):
    assert len(x.size()) == 3
    B, N, D = x.size()
    assert_mean_zero(x)

    # r is invariant to a basis change in the relevant hyperplane.
    r2 = sum_except_batch(x.pow(2))

    # The relevant hyperplane is (N-1) * D dimensional.
    degrees_of_freedom = (N-1) * D

    # Normalizing constant and logpx are computed:
    log_normalizing_constant = -0.5 * degrees_of_freedom * np.log(2*np.pi)
    log_px = -0.5 * r2 + log_normalizing_constant

    return log_px


def sample_center_gravity_zero_gaussian(size, device):
    assert len(size) == 3
    x = torch.randn(size, device=device)

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = remove_mean(x)
    return x_projected


def center_gravity_zero_gaussian_log_likelihood_with_mask(x, node_mask):
    assert len(x.size()) == 3
    B, N_embedded, D = x.size()
    assert_mean_zero_with_mask(x, node_mask)

    # r is invariant to a basis change in the relevant hyperplane, the masked
    # out values will have zero contribution.
    r2 = sum_except_batch(x.pow(2))

    # The relevant hyperplane is (N-1) * D dimensional.
    N = node_mask.squeeze(2).sum(1)  # N has shape [B]
    degrees_of_freedom = (N-1) * D

    # Normalizing constant and logpx are computed:
    log_normalizing_constant = -0.5 * degrees_of_freedom * np.log(2*np.pi)
    log_px = -0.5 * r2 + log_normalizing_constant

    return log_px


def sample_center_gravity_zero_gaussian_with_mask(size, device, node_mask):
    assert len(size) == 3
    x = torch.randn(size, device=device)

    x_masked = x * node_mask

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = remove_mean_with_mask(x_masked, node_mask)
    return x_projected


def standard_gaussian_log_likelihood(x):
    # Normalizing constant and logpx are computed:
    log_px = sum_except_batch(-0.5 * x * x - 0.5 * np.log(2*np.pi))
    return log_px


def sample_gaussian(size, device):
    x = torch.randn(size, device=device)
    return x


def standard_gaussian_log_likelihood_with_mask(x, node_mask):
    # Normalizing constant and logpx are computed:
    log_px_elementwise = -0.5 * x * x - 0.5 * np.log(2*np.pi)
    log_px = sum_except_batch(log_px_elementwise * node_mask)
    return log_px


def sample_gaussian_with_mask(size, device, node_mask):
    x = torch.randn(size, device=device)
    x_masked = x * node_mask
    return x_masked
