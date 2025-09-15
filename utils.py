import numpy as np
import getpass
import os
import torch

# Folders
def create_folders(args):
    try:
        os.makedirs('outputs')
    except OSError:
        pass

    try:
        os.makedirs('outputs/' + args.exp_name)
    except OSError:
        pass


# Model checkpoints
def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


#Gradient clipping
class Queue():
    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)

def gradient_clipping_2(flow, gradnorm_queue, safety_max=1500.0):
    # Calculate adaptive threshold with more generous multipliers
    if len(gradnorm_queue) > 10:  # Need more history for reliable stats
        adaptive_max = 2.0 * gradnorm_queue.mean() + 3.0 * gradnorm_queue.std()
        # Safety net to prevent completely runaway gradients
        max_grad_norm = min(adaptive_max, safety_max)
    else:
        # During warmup, be very permissive
        max_grad_norm = safety_max
    
    grad_norm = torch.nn.utils.clip_grad_norm_(
        flow.parameters(), max_norm=max_grad_norm, norm_type=2.0
    )
    
    # Add to queue (maybe with some outlier protection)
    if grad_norm < safety_max * 2:  # Only exclude truly extreme outliers
        gradnorm_queue.add(float(grad_norm))
    
    if grad_norm > max_grad_norm:
        print(f'Clipped gradient: {grad_norm:.1f} -> {max_grad_norm:.1f}')
    
    return grad_norm
    

def gradient_clipping(flow, gradnorm_queue):
    # Allow gradient norm to be 150% + 2 * stdev of the recent history.
    max_grad_norm = 1.5 * gradnorm_queue.mean() + 2 * gradnorm_queue.std()

    # Clips gradient and returns the norm
    grad_norm = torch.nn.utils.clip_grad_norm_(
        flow.parameters(), max_norm=max_grad_norm, norm_type=2.0)

    if float(grad_norm) > max_grad_norm:
        gradnorm_queue.add(float(max_grad_norm))
    else:
        gradnorm_queue.add(float(grad_norm))

    if float(grad_norm) > max_grad_norm:
        print(f'Clipped gradient with value {grad_norm:.1f} '
              f'while allowed {max_grad_norm:.1f}')
    return grad_norm


# Rotation data augmntation
def random_rotation(x):
    bs, n_nodes, n_dims = x.size()
    device = x.device
    angle_range = np.pi * 2
    if n_dims == 2:
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        R_row0 = torch.cat([cos_theta, -sin_theta], dim=2)
        R_row1 = torch.cat([sin_theta, cos_theta], dim=2)
        R = torch.cat([R_row0, R_row1], dim=1)

        x = x.transpose(1, 2)
        x = torch.matmul(R, x)
        x = x.transpose(1, 2)

    elif n_dims == 3:

        # Build Rx
        Rx = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Rx[:, 1:2, 1:2] = cos
        Rx[:, 1:2, 2:3] = sin
        Rx[:, 2:3, 1:2] = - sin
        Rx[:, 2:3, 2:3] = cos

        # Build Ry
        Ry = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Ry[:, 0:1, 0:1] = cos
        Ry[:, 0:1, 2:3] = -sin
        Ry[:, 2:3, 0:1] = sin
        Ry[:, 2:3, 2:3] = cos

        # Build Rz
        Rz = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Rz[:, 0:1, 0:1] = cos
        Rz[:, 0:1, 1:2] = sin
        Rz[:, 1:2, 0:1] = -sin
        Rz[:, 1:2, 1:2] = cos

        x = x.transpose(1, 2)
        x = torch.matmul(Rx, x)
        #x = torch.matmul(Rx.transpose(1, 2), x)
        x = torch.matmul(Ry, x)
        #x = torch.matmul(Ry.transpose(1, 2), x)
        x = torch.matmul(Rz, x)
        #x = torch.matmul(Rz.transpose(1, 2), x)
        x = x.transpose(1, 2)
    else:
        raise Exception("Not implemented Error")

    return x.contiguous()


# Other utilities
def get_wandb_username(username):
    if username == 'cvignac':
        return 'cvignac'
    current_user = getpass.getuser()
    if current_user == 'victor' or current_user == 'garciasa':
        return 'vgsatorras'
    else:
        return username

# diagnosing/testing function used during developmen
def normalize(x, h, node_mask):
    # compute 
    node_mask = node_mask.float()
    node_mask_unsqueezed = node_mask.unsqueeze(-1)  # (B, N, 1)
    
    # re-center -> done in remove_mean_with_mask (I believe).
    # mean = (x * node_mask_unsqueezed).sum(dim=1, keepdim=True) / node_mask_unsqueezed.sum(dim=1, keepdim=True).clamp(min=1)
    # x = x - mean

    # re-scale (->normalize)
    var = ((x**2) * node_mask_unsqueezed).sum(dim=1, keepdim=True) / node_mask_unsqueezed.sum(dim=1, keepdim=True).clamp(min=1)
    std = var.sqrt().clamp(min=1e-3)
    x = x / std

    # we dont normalize categorical for now..
    h_cat = h['categorical'].float()

    # we are going to apply the node mask to h integers and normalize it by its global biggest value
    h_int_raw = h['integer'].float()  # shape (B, N)
    
    # node mask (no padding/CLS/asterisk tokens included) (only 'actual'/'physical' atoms)
    atom_mask = (h_int_raw > 0).float() * node_mask  # (B, N)

    # re-scale by dividing by largest charge in dataset
    h_int_norm = (h_int_raw / 20.0) * atom_mask

    return x, {'categorical': h_cat, 'integer': h_int_norm}


if __name__ == "__main__":


    ## Test random_rotation
    bs = 2
    n_nodes = 16
    n_dims = 3
    x = torch.randn(bs, n_nodes, n_dims)
    print(x)
    x = random_rotation(x)
    #print(x)
