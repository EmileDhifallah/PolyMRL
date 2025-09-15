# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import random
# import os
import copy
import utils
import json
import argparse
from pathlib import Path
from datetime import datetime

import wandb
from configs.datasets_config import get_dataset_info
from os.path import join
from pi1m import dataset
from pi1m.models import get_optim, get_model
from pi1m.utils import compute_mean_mad_from_dataloader
from helpers_from_edm import en_diffusion
from helpers_from_edm.utils import assert_correctly_masked
from helpers_from_edm import utils as flow_utils
from plot_lines import plot_prediction_logs

import torch
from torch.cuda.amp import autocast, GradScaler

import time
import os
import pickle
import numpy as np
from pi1m.utils import prepare_context, compute_mean_mad
from pi1m.models import DistributionProperty, DistributionNodes
from train_test import train_epoch, test, analyze_and_save
from model.model_classes import PolyMRL

parser = argparse.ArgumentParser(description='E3Diffusion')
parser.add_argument('--exp_name', type=str, default='debug_10')
parser.add_argument('--model', type=str, default='egnn_dynamics',
                    help='our_dynamics | schnet | simple_dynamics | '
                         'kernel_dynamics | egnn_dynamics |gnn_dynamics')
parser.add_argument('--probabilistic_model', type=str, default='diffusion',
                    help='diffusion')
# Training complexity is O(1) (unaffected), but sampling complexity is O(steps).
parser.add_argument('--diffusion_steps', type=int, default=500)
parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                    help='learned, cosine')
parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5,
                    )
parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                    help='vlb, l2')

parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--brute_force', type=eval, default=False,
                    help='True | False')
parser.add_argument('--actnorm', type=eval, default=True,
                    help='True | False')
parser.add_argument('--break_train_epoch', type=eval, default=False,
                    help='True | False')
parser.add_argument('--dp', type=eval, default=True,
                    help='True | False')
parser.add_argument('--condition_time', type=eval, default=True,
                    help='True | False')
parser.add_argument('--clip_grad', type=eval, default=True,
                    help='True | False')
parser.add_argument('--trace', type=str, default='hutch',
                    help='hutch | exact')
# EGNN args -->
parser.add_argument('--n_layers', type=int, default=6,
                    help='number of layers')
parser.add_argument('--inv_sublayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--nf', type=int, default=128,
                    help='number of layers')
parser.add_argument('--tanh', type=eval, default=True,
                    help='use tanh in the coord_mlp')
parser.add_argument('--attention', type=eval, default=True,
                    help='use attention in the EGNN')
parser.add_argument('--norm_constant', type=float, default=1,
                    help='diff/(|diff| + norm_constant)')
parser.add_argument('--sin_embedding', type=eval, default=False,
                    help='whether using or not the sin embedding')
# <-- EGNN args
parser.add_argument('--weight_decay', type=float, default=1e-3,
                    help='the weight decay parameter to pass to the (AdamW) optimizer')
parser.add_argument('--ode_regularization', type=float, default=1e-3)
parser.add_argument('--dataset', type=str, default='Eat',
                    help='qm9 | qm9_second_half (train only on the last 50K samples of the training dataset)')
parser.add_argument('--datadir', type=str, default='qm9/temp',
                    help='qm9 directory')
parser.add_argument('--filter_n_atoms', type=int, default=None,
                    help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
parser.add_argument('--dequantization', type=str, default='argmax_variational',
                    help='uniform | variational | argmax_variational | deterministic')
parser.add_argument('--n_report_steps', type=int, default=1)
parser.add_argument('--wandb_usr', type=str)
parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
parser.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--save_model', type=eval, default=True,
                    help='save model')
parser.add_argument('--generate_epochs', type=int, default=1,
                    help='save model')
parser.add_argument('--num_workers', type=int, default=0, help='Number of worker for the dataloader')
parser.add_argument('--test_epochs', type=int, default=10)
parser.add_argument('--data_augmentation', type=eval, default=False, help='use attention in the EGNN')
parser.add_argument("--conditioning", nargs='+', default=[],
                    help='arguments : homo | lumo | alpha | gap | mu | Cv' )
parser.add_argument('--resume', type=str, default=None,
                    help='')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='')
parser.add_argument('--ema_decay', type=float, default=0.999,
                    help='Amount of EMA decay, 0 means off. A reasonable value'
                         ' is 0.999.')
parser.add_argument('--augment_noise', type=float, default=0)
parser.add_argument('--n_stability_samples', type=int, default=500,
                    help='Number of samples to compute the stability') # are these samples saved to device or just used for calculating the stability and then discarded?
parser.add_argument('--normalize_factors', type=eval, default=[1, 4, 1],
                    help='normalize factors for [x, categorical, integer]')
parser.add_argument('--remove_h', action='store_true')
parser.add_argument('--include_charges', type=eval, default=True,
                    help='include atom charge or not')
parser.add_argument('--visualize_every_batch', type=int, default=1e8,
                    help="Can be used to visualize multiple times per epoch")
parser.add_argument('--normalization_factor', type=float, default=1,
                    help="Normalize the sum aggregation of EGNN")
parser.add_argument('--aggregation_method', type=str, default='sum',
                    help='"sum" or "mean"')
parser.add_argument('--train_mode', type=str, default='property_prediction',
                    help='if we are doing "pretrain" or "property_prediction"')
parser.add_argument('--freeze_encoders', type=eval, default=True,
                    help='freeze encoders during downstream prediction - or keep training their params.')
parser.add_argument('--freeze_projections', type=eval, default=False,
                    help='freeze projection layers (contr. module) during downstream prediction - or keep training their params.')
parser.add_argument('--pretrained_dir', type=str, default='outputs/polyMRL_qm9_test',
                    help='directory in which pretrained encoder/layer weights are stored')
parser.add_argument('--mixed_precision_training', type=eval, default=True,
                    help='whether to turn on torch AutoCast plus gradient scaler for FP16 (MP) training')
parser.add_argument('--cls_rep_3d', type=str, default='virtual_atom',
                    help='virtual_atom|naive|mean_pool|max_pool; way to extract 3D final atom repr. for downstream finetuning/prediction (like CLS in LM).')
parser.add_argument('--reflection_equiv', type=eval, default=True,
                    help='whether to enable reflection equivariance in the equivariant (coordinate) updates, or to avoid it (DiffSBDD avoids it, so "False")')
parser.add_argument('--amp_dtype', type=str, default='16',
                    help='16|32 which torch float type to use within amp training')
parser.add_argument('--pretrain_checkpoint_dir_1d', type=str, default=None,
                    help='dir to checkpoint containing a pretrained 1d encoder')
parser.add_argument('--pretrain_checkpoint_dir_3d', type=str, default=None,
                    help='dir to checkpoint containing a pretrained 3d encoder')
parser.add_argument('--unscale', type=eval, default=True,
                    help='whether to enable grad unscaling in amp')
# following args for prediction finetuning
parser.add_argument('--variable_lrs', type=eval, default=False,
                    help='in property prediction, whether to set lr higher for the class. head than for the encoders')                    
parser.add_argument('--unfreeze_encoders', type=eval, default=False,
                    help='in property prediction, whether to unfreeze encoders after some initial iterations')
parser.add_argument('--unfreeze_projections', type=eval, default=False,
                    help='in property prediction, whether to unfreeze the projection layer(s) after some initial iterations')
parser.add_argument('--use_layernorm', type=eval, default=False,
                    help='in property prediction, whether to utilize Layernorm before the classification head (normalizes cls_representation to (0,1))')
parser.add_argument('--only_1d', type=eval, default=False,
                    help='in property prediction, whether to set cls_representation to purely the 1d encoder output')
parser.add_argument('--only_3d', type=eval, default=False,
                    help='in property prediction, whether to set cls_representation to purely the 3d encoder output')
parser.add_argument('--scaling_value', type=float, default=1.,
                    help="value by which we multiply property predictions to scale to target distribution; default of 1. means initialization of 1.0 (no intial effect, but might adapt during learning))")
parser.add_argument('--oligomer', type=eval, default=False,
                    help="whether we are using oligomers (or monomers instead)")

args = parser.parse_args()

print("args list: ", args)

dataset_info = get_dataset_info(args.dataset, args.remove_h, args.cls_rep_3d)

atom_encoder = dataset_info['atom_encoder'] # dict like {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}; I believe for generating mols, but it does not align with the used charges input
atom_decoder = dataset_info['atom_decoder']
max_charge = max(atom_encoder.values()) # get max charge value e.g. 7 or 29

# args, unparsed_args = parser.parse_known_args()
args.wandb_usr = utils.get_wandb_username(args.wandb_usr)

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32

if args.resume is not None:
    exp_name = args.exp_name + '_resume'
    start_epoch = args.start_epoch
    resume = args.resume # if we resume training from a previous checkpoint/pytorch state dict
    wandb_usr = args.wandb_usr
    normalization_factor = args.normalization_factor
    aggregation_method = args.aggregation_method # is this for the (last) layer of the GNN (neighbourhood aggregation)?

    with open(join(args.resume, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    args.resume = resume
    args.break_train_epoch = False # so if we resume model, we do not break train epoch

    args.exp_name = exp_name
    args.start_epoch = start_epoch
    args.wandb_usr = wandb_usr

    # Careful with this -->
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = normalization_factor
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = aggregation_method

    print(args)

utils.create_folders(args) # create outputs/exp_name folder
# print(args)

# setting torch, random and np seeds to the same value
def set_seed(seed: int = 42):
    # Python built-in RNG
    random.seed(seed)

    # NumPy RNG
    np.random.seed(seed)

    # Torch RNG (CPU and CUDA)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

set_seed(42)

# Wandb config
if args.no_wandb:
    mode = 'disabled'
else:
    mode = 'online' if args.online else 'offline'
kwargs = {'entity': args.wandb_usr, 'name': args.exp_name, 'project': 'e3_diffusion', 'config': args,
          'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': 'offline'}
wandb.init(**kwargs)
wandb.save('*.txt')

# Retrieve QM9 dataloaders --> this also downloads the QM9 dataset!
# note: check how batch is handled in the dataset class/dataloader (data collator) now, and how the smiles is handled; we need to return the smiles somehow (check how we need it).
dataloaders, charge_scale, tokenizer = dataset.retrieve_dataloaders(args) # what is the charge scale?
print('loaders retrieved')

data_dummy = next(iter(dataloaders['train'])) # what is the data_dummy used for (seems like a single batch of training data for testing something)?
print('data dummy: ', data_dummy)
print('done') # we are getting a data dummy from .lmdb file; that means we now have to take care of the class. head, the loss signal (e.g. no MLM labels during training?), and how the encoders should be frozen during training.


# get size of atom vocab from one hot vector of the data dummy (i.e. the emb. size of h)
# in_node_nf = data_dummy['one_hot'].shape[-1]
in_node_nf = len(atom_decoder) + int(args.include_charges) # atom_decoder is a dict of the heavy atoms present in the dataset; thus, the length of the atom vocabulary; for qm9, len(vocab)=4, 5 including charges (, 6 also including time-conditioning)

train_target_mad_mean={}
# if we are in downstream property prediction, use PI1M (pretrain data)'s atom vocabulary to match h-in-embedding sizes
if args.train_mode=='property_prediction':
    pi1m_info = get_dataset_info('PI1M', args.remove_h, 'pretrain')
    in_node_nf = len(pi1m_info['atom_decoder']) + int(args.include_charges)

    # we also need the train set's mad if we are doing downstream property prediction
    train_target_mad_mean={}
    property_norms = compute_mean_mad_from_dataloader(dataloaders['train'], ['target'])
    train_target_mad_mean['mad'] = property_norms['target']['mad']
    train_target_mad_mean['mean'] = property_norms['target']['mean']


# note: leave this in __main__ flow for now, think this is nice to keep to create/keep functionality for adding a property as context to our model
# check if conditional generation is switched on, if so calculate stats of molecule properties (for what reason?) and context dummy (for what reason?)
# how do we condition on a specific property?
prop_dist = None
if len(args.conditioning) > 0:
    print(f'Conditioning on {args.conditioning}')
    property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
    context_dummy = prepare_context(args.conditioning, data_dummy, property_norms)
    context_node_nf = context_dummy.size(2)

    # in case we want to condition on a molecule property!  
    # create a numeric distribution of the property condition(s) that  we are adding
    prop_dist = DistributionProperty(dataloaders['train'], args.conditioning) # I think to get the distribution for the denoising model?
    
else:
    context_node_nf = 0 # 0 context nodes if no conditional generation
    property_norms = None

args.context_node_nf = context_node_nf


# Create EGNN flow --> what does 'flow' even mean here?
# note: I think we can simply initialize EGNN with required arguments, no need to use get_model entirely; alternatively
# we can also define the model by making use of their get_model function, to play it safe

# -> update: do not need other functions of get model for this instance

# model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloaders['train'])
# if prop_dist is not None: # this is None if no conditional generation
    # prop_dist.set_normalizer(property_norms) # if conditional generation, normalize molecule property distribution

# 1d setup

model = polyMRL(in_node_nf=in_node_nf, context_node_nf=context_node_nf,
                 n_dims=3, hidden_nf=64, device=device, tokenizer=tokenizer,
                 train_mode=args.train_mode, freeze_encoders=args.freeze_encoders,
                 freeze_projections=args.freeze_projections, pretrained_dir=args.pretrained_dir,
                 cls_rep_3d=args.cls_rep_3d, reflection_equiv=args.reflection_equiv,
                 pretrain_checkpoint_dir_1d=args.pretrain_checkpoint_dir_1d, pretrain_checkpoint_dir_3d=args.pretrain_checkpoint_dir_3d,
                 norm_values=max_charge, use_layernorm=args.use_layernorm,
                 only_1d=args.only_1d, only_3d=args.only_3d, scaling_value=args.scaling_value, oligomer=args.oligomer)
                # not using args right now: 
                # act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                #  condition_time=True, tanh=False, mode='egnn_dynamics', norm_constant=0,
                #  inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum',
                   # remove/fix this as well
# model_reg_copy = model.

model = model.to(device)
# define AdamW optimizer for model
optim = get_optim(args, model) # load AdamW optimizer with model's initial weight matrix/array and hyperparams
print('model: ', model)
print('optim: ', optim)

gradnorm_queue = utils.Queue() # set up queue to continuously update grad normalization values over time during training (right?)
gradnorm_queue.add(3000)  # Add large value that will be flushed.

# gradient-scaler to compensate for mixed precision training
# note: this scaler is either disabled or enabled depending on the enabled arg, set in our args list;
device_type = "cuda" if args.cuda else "cpu"
scaler = torch.amp.GradScaler(device_type, enabled=args.mixed_precision_training)


def check_mask_correct(variables, node_mask): # does masking mean noising of atoms maybe? What else could masking mean in the diffusions setting?
    for variable in variables:
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def main():

    if args.resume is not None: # resume training from state dict if given in arguments
        flow_state_dict = torch.load(join(args.resume, 'flow.npy')) # what is the flow state dict?
        optim_state_dict = torch.load(join(args.resume, 'optim.npy'))
        model.load_state_dict(flow_state_dict)
        optim.load_state_dict(optim_state_dict)
        print('model state dict: ', model.load_state_dict)

    # can stay for now but likely will not use parallel computing for current model iteration.
    # Initialize dataparallel if enabled and possible.
    print('nr of devices: ', torch.cuda.device_count())
    print('name device 0, torch.cuda.get_device_name(0):', torch.cuda.get_device_name(0))

    if args.dp and torch.cuda.device_count() > 1:
        print(f'Training using {torch.cuda.device_count()} GPUs')
        model_dp = torch.nn.DataParallel(model.cpu())
        model_dp = model_dp.cuda()
    else:
        model_dp = model
    print('model, model_dp: ', model, model_dp)

    # can stay I think?
    # Initialize model copy for exponential moving average of params. --> what is EMA of params used/beneficial for?
    
    # if ema is enabled, we make a copy of the model and define ema appropriately.
    # if dp is also enabled, we put this model copy on parallel nodes
    if args.ema_decay > 0:
        model_ema = copy.deepcopy(model) # why do we need a separate saved model for this ema version?
        ema = flow_utils.EMA(args.ema_decay) # store ema values

        if args.dp and torch.cuda.device_count() > 1:
            model_ema_dp = torch.nn.DataParallel(model_ema)
        else:
            model_ema_dp = model_ema
    # if ema is not enabled, then we simply set the ema model to the original model instance, and the dp version to the dp version of the model, which is simply the original model instance if dp is not enabled.
    else:
        ema = None # no ema setup, if not given in args that ema_decay is utilized
        model_ema = model # why do we still initialize a model_ema model if we do not utilize it though?
        model_ema_dp = model_dp


    # initialize loss values (as very big number); best val loss is used to select best validation epoch's model for testing; best_nll_test is the testing epoch where we find the model's minimal test loss
    best_nll_val = 1e8
    best_nll_test = 1e8

    nll_vals = []
    nll_tests = []
    grad_vals = {epoch:{} for epoch in range(args.start_epoch, args.n_epochs)}

    # added myself from pi1m/models.py
    histogram = dataset_info['n_nodes']
    nodes_dist = DistributionNodes(histogram)

    # batch losses-logging
    loss_logging_dict = {}

    if args.train_mode=='property_prediction':
        assert args.pretrained_dir is not None
        # args.pretrained_dir
        checkpoint = torch.load(os.path.join("outputs", "final_model_1d_PI1M-weightdecay-no_amp-fp32_epoch_4", "pretrained_encoders_proj_heads.pth"), map_location=device_type) # entire pretrained-model's state_dict
        net_1d_pretrained_weights = checkpoint["encoder_1d"] # pretrained and aligned 1d encoder if from combined pretrain model; if from 1d trained model only, this would be simply a 1d encoder (pretrained LM)
        # model.net_1d.load_state_dict(checkpoint["encoder_1d"], strict=False)

        # take only the 1d encoders from checkpoint for the reg. test.

    for epoch in range(args.start_epoch, args.n_epochs):
        print(f'epoch {epoch} started')
        start_epoch = time.time()
        print("args=args, loader=dataloaders['train'], epoch=epoch, model=model, model_dp=model_dp,model_ema=model_ema, ema=ema, device=device, dtype=dtype, property_norms=property_norms,nodes_dist=nodes_dist, dataset_info=dataset_info,gradnorm_queue=gradnorm_queue, optim=optim, prop_dist=prop_dist")
        print(args, dataloaders['train'], epoch,# model, model_dp,model_ema, ema, 
                    device, dtype, property_norms,
                    nodes_dist, dataset_info,
                    gradnorm_queue, optim, prop_dist)

        model.partition = 'train'
        model_dp.partition = 'train'
        train_epoch(args=args, loader=dataloaders['train'], epoch=epoch, model=model, model_dp=model_dp,
                    model_ema=model_ema, ema=ema, device=device, dtype=dtype, property_norms=property_norms,
                    nodes_dist=nodes_dist, dataset_info=dataset_info,
                    gradnorm_queue=gradnorm_queue, optim=optim, prop_dist=prop_dist, grad_scaler=scaler,
                    loss_logging_dict=loss_logging_dict, train_target_mad_mean=train_target_mad_mean, grad_vals=grad_vals, net_1d_pretrained_weights=net_1d_pretrained_weights) # start training epoch i
        print(f"Epoch took {time.time() - start_epoch:.1f} seconds.")
        # now, go and write/modify the training loop (like train_epoch from train_test.py) -> remove things, add things from list (in model_classes.py).
        # also: add any processing and transformations needed for the LM to work/for the language input to be succesfully passed to the model forward (and into the LM).

        # we have now performed the training epoch; in this epoch, no validating or testing takes place.
        # instead, below here, we perform validation and testing functionality/cycle.

        # if we are on a test epoch number, we perform testing, analyzing and saving models/metrics (and logging to wandb);
        # -> we set test_epochs to 1, so that we run valid loop every epoch, and then test one time using the best model (or every epoch, so we can check how it changes per epoch.);

        # code starts here
        if epoch % args.test_epochs == 0: # yes
            # log noising (gamma values) parameters/information into wandb, but only if model is egnn diffusion model (I think?)
            # if isinstance(model, polyMRL):    #  (en_diffusion.EnVariationalDiffusion):
            #     wandb.log(model.log_info(), commit=True)
            
            # computes stability/etc metrics for the given epoch and training model instance;
            # metrics are logged to wandb and returned
            # note: we do not update node distribution (unlike DDPM), so no sampling taking place for now (which is the core of the analyze_and_save() func.);
            # if not args.break_train_epoch: # if not breaking epoch, e.g. when resuming training from checkpoint?
            #     analyze_and_save(args=args, epoch=epoch, model_sample=model_ema, nodes_dist=nodes_dist, # stability, novelty, etc..
            #                      dataset_info=dataset_info, device=device,
            #                      prop_dist=prop_dist, n_samples=args.n_stability_samples)

            
            # perform validation epoch, mean val loss stored in nll_val; ema-model is used for (hopefully) better generalization to the valid/test sets;
            # nll_val from 'test' function is the average loss value per molecule in the test epoch
            model.partition = 'valid'
            model_dp.partition = 'valid'
            model_ema_dp.partition = 'valid'
            nll_val = test(args=args, loader=dataloaders['valid'], epoch=epoch, eval_model=model_ema_dp,
                           partition='Val', device=device, dtype=dtype, nodes_dist=nodes_dist,
                           property_norms=property_norms, loss_logging_dict=loss_logging_dict,
                           train_target_mad_mean=train_target_mad_mean)
            nll_vals+=[nll_val]
            
            # perform test epoch, mean (/mol) test loss stored in 'nll_test';
            # this means that we run a test epoch during each training epoch;
            # this is unnecessary, as we only wish to run the test set during the best of
            # the validation epochs' performance; however, it brings more ease of use and reading this code,
            # and the test epochs do not take that long so it will not make this script that much more inefficient;
            model.partition = 'test'
            model_dp.partition = 'test'
            model_ema_dp.partition = 'test'
            nll_test = test(args=args, loader=dataloaders['test'], epoch=epoch, eval_model=model_ema_dp,
                            partition='Test', device=device, dtype=dtype, nodes_dist=nodes_dist,
                            property_norms=property_norms, loss_logging_dict=loss_logging_dict,
                            train_target_mad_mean=train_target_mad_mean)
            nll_tests+=[nll_test]

            # now we check if the current epoch's validation loss is the lowest valid. loss found so far; if so, it means that this is our currently best model;
            # which entails that we would like to run our test epoch on this epoch's model as well. Hence, we take the current test loop's
            # average loss as the best/final test loss so far. At the end of n_epochs, we will save the test loss from the epoch
            # where we achieved the highest overall (average) validation loss;
            if nll_val < best_nll_val: # if current val performance is the best yet, store current val and test accs. (nll values)
                best_nll_val = nll_val
                best_nll_test = nll_test

                # if args.save_model: # we also save model of the current epoch (so not only if it is the best-yet version of the model)

                if args.train_mode!='property_prediction':
                    # save model weights if we are at best-yet validation performance
                    utils.save_model(optim, 'outputs/%s/optim_best_val.npy' % (args.exp_name)) # instead of %d, %(epoch)
                    utils.save_model(model, 'outputs/%s/generative_model_best_val.npy' % (args.exp_name))

                    # also again the ema version if applicable
                    if args.ema_decay > 0:
                        utils.save_model(model_ema, 'outputs/%s/generative_model_ema_best_val.npy' % (args.exp_name))
                    # and the args again
                    with open('outputs/%s/args_best_val.pickle' % (args.exp_name), 'wb') as f:
                        pickle.dump(args, f)

                    # save encoders for contrastive learning testing
                    torch.save(model.net_1d.state_dict(), f"outputs/{args.exp_name}/encoder_1d.pth")
                    torch.save(model.net_dynamics.state_dict(), f"outputs/{args.exp_name}/encoder_3d.pth")

                    # do this every epoch
                    save_dict = {
                    "encoder_1d": model.net_1d.state_dict(),
                    "encoder_3d": model.net_dynamics.state_dict(),
                    "gamma_network": model.gamma.state_dict(),
                    "contr_1d_layer": model.contr_1d_layer.state_dict(),
                    "contr_3d_layer": model.contr_3d_layer.state_dict(),
                    "contr_3d_layer_xh": model.contr_3d_layer_xh.state_dict(),
                    }
                
                    torch.save(save_dict, f"outputs/{args.exp_name}/pretrained_encoders_proj_heads.pth")


            # do this every epoch
            save_dict = {
            "encoder_1d": model.net_1d.state_dict(),
            "encoder_3d": model.net_dynamics.state_dict(),
            "gamma_network": model.gamma.state_dict(),
            "contr_1d_layer": model.contr_1d_layer.state_dict(),
            "contr_3d_layer": model.contr_3d_layer.state_dict(),
            "contr_3d_layer_xh": model.contr_3d_layer_xh.state_dict(),
            "contr_3d_layer_hidden": model.contr_3d_layer_hidden.state_dict(),
            }

            if args.train_mode != 'property_prediction':
                # create exp_name_epoch directory if its not there yet, then save current epoch's torch weights
                Path(f"outputs/{args.exp_name}_epoch_{epoch}").mkdir(parents=True, exist_ok=True)
                torch.save(save_dict, f"outputs/{args.exp_name}_epoch_{epoch}/pretrained_encoders_proj_heads.pth")



            ## note: commented analyze/sample, and testing functinalities out until we implement test loop and/or analysis function!
            # w.r.t. analyze_and_save: every epoch, we carry out some stability and other metrics (e.g. novelty) calculations on a set of generated molecules,
            # and save these to our device; the samples are generated using our urrent learned training data dist
            
            
            ## old; (basically the same as saving lines above besides 'last' in path name)
            #     # save current (best) model and optimizer params.
            #     if args.save_model: 
            #         args.current_epoch = epoch + 1
            #         utils.save_model(optim, 'outputs/%s/optim.npy' % args.exp_name)
            #         utils.save_model(model, 'outputs/%s/generative_model.npy' % args.exp_name)
                    
            #         # if ema on, save ema model too
            #         if args.ema_decay > 0:
            #             utils.save_model(model_ema, 'outputs/%s/generative_model_ema.npy' % args.exp_name)
                    
            #         # save used args for our model training too
            #         with open('outputs/%s/args.pickle' % args.exp_name, 'wb') as f:
            #             pickle.dump(args, f)


            # print out current epoch's mean val and test losses, and log these together with current best test loss value to wandb
            print('Val loss: %.4f \t Test loss:  %.4f' % (nll_val, nll_test))
            print('Best val loss so far: %.4f \t Best test loss so far:  %.4f' % (best_nll_val, best_nll_test))

            # log epoch's mean val and test losses to wandb
            wandb.log({"Val loss ": nll_val}, commit=True)
            wandb.log({"Test loss ": nll_test}, commit=True)
            wandb.log({"Best cross-validated test loss ": best_nll_test}, commit=True)

        
        # save current loss logging dict as json; only save last epoch for poperty prediction runs
        if args.train_mode=='pretrain' or (args.train_mode=='property_prediction' and (epoch%1000==0 or epoch==(args.n_epochs-1))): # epoch==(args.n_epochs-1))
            save_path_inter = join('output_files', args.train_mode+'_cv', 'loss_dicts', f'iter_loss_dict_{args.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_epoch_{epoch}.json')
            with open(save_path_inter, 'w', encoding='utf-8') as f:
                json.dump(loss_logging_dict, f, ensure_ascii=False, indent=4)
            
            print(f'Saving plot epoch {epoch}')
            plot_prediction_logs(file_path=save_path_inter,#final_save_name,
                                 labels=('Prediction-variability', 'Train', 'Val', 'Test', 'Test-RMSE'),
                                 title=f"Experiment: {args.exp_name} -- train and evaluation scores and std per iteration's predictions",
                                 palette='muted',
                                 save_path=f'output_files/property_prediction/plots/plot_predictionvals-{args.exp_name}_epoch_{epoch}.png',
                                 drop_iters=True,
                                 
                                )

        # save gradient vals dict
        if (epoch%1000==0 or epoch==(args.n_epochs-1)): # epoch==(args.n_epochs-1)
            print(f'epoch {epoch}, saving gradient vals..')
            with open(join('output_files', args.train_mode+'_cv', 'gradient_vals', f'gradient_vals_{args.exp_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}_epoch_{epoch}.json'), 'w', encoding='utf-8') as f:
                json.dump(grad_vals, f, ensure_ascii=False, indent=4)
            

    # printing for overview in log/output file
    print('All mean validation losses (1 value per epoch): ', nll_vals)
    print('All mean test losses (1 value per epoch): ', nll_tests)

    # saving all batch losses to JSON file to read in for graph plotting
    # NOTE: commented this for CV run!
    final_save_name = join('output_files', args.train_mode+'_cv', 'loss_dicts', f'iter_loss_dict_{args.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json')

    with open(final_save_name, 'w', encoding='utf-8') as f:
        json.dump(loss_logging_dict, f, ensure_ascii=False, indent=4)

    # if property prediction, plot the prediction values together with the experiment's loss values directly;
    if args.train_mode=='property_prediction':
        if 'Property_predictions' in loss_logging_dict.keys():
            plot_prediction_logs(file_path=final_save_name,
                                 labels=('Prediction-variability', 'Train', 'Val', 'Test', 'Test-RMSE'),
                                 title=f"{args.exp_name} - Standard deviation per iteration's regression predictions, together with train/valid/test losses",
                                 palette='muted',
                                 save_path=f'output_files/property_prediction/plots/plot_predictionvals-{args.exp_name}_last.png',
                                 drop_iters=True,
                                 
                                )
        else:
            print('WARNING: property prediction values not present in log dict! No plot was saved to device')

            
if __name__ == "__main__":
    main()



'''
data dummy shape:

{'1d': {'input_ids': tensor([[  0, 262, 263,  ...,   1,   1,   1],
        [  0, 262,  12,  ...,   1,   1,   1],
        [  0, 309,  21,  ...,   1,   1,   1],
        ...,
        [  0,  51,  33,  ...,   1,   1,   1],
        [  0, 262,  21,  ...,   1,   1,   1],
        [  0, 282,  21,  ...,   1,   1,   1]]), 'attention_mask': tensor([[1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        ...,
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0]])},
 '3d': {'num_atoms': tensor([14, 19, 15, 15, 21,  9, 18, 14, 17, 19, 18, 10, 19, 20, 21, 19, 16, 16,
        20, 12, 18, 16, 18, 17, 17, 17, 19, 17, 19, 13, 17, 14, 24, 16, 17, 15,
        19, 21, 15, 21, 17, 20, 20, 18, 19, 15, 12, 20, 24, 21, 17, 25, 11, 16,
        21, 23, 21, 23, 18, 17, 21, 13, 19, 19]), 'charges': tensor([[[6], ...,]]),
        ...}
}
'''