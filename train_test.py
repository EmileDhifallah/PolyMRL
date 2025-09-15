import wandb
from helpers_from_edm.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked, sample_center_gravity_zero_gaussian_with_mask
import numpy as np
import pi1m.visualizer as vis
from pi1m.analyze import analyze_stability_for_molecules
from pi1m.sampling import sample_chain, sample, sample_sweep_conditional
import utils
import pi1m.utils as qm9utils
from pi1m import losses
import time
from datetime import datetime
import torch
import json

def print_diagnostics(tens_name, inp_tens):
    with torch.no_grad():
        print(f"{tens_name} stats:")
        print("  norm:", inp_tens.norm().item())
        print("  min:", inp_tens.min().item())
        print("  max:", inp_tens.max().item())
        print("  mean:", inp_tens.mean().item())
        print("  std:", inp_tens.std().item())
        
def check_infs(model, model_name="EGNN"):
    inf_check=False
    # print(f"\n--- Gradients for {model_name} ---")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_min = param.grad.min().item()
            grad_max = param.grad.max().item()
            has_nan = torch.isnan(param.grad).any().item()
            # print(f"{name}: norm={grad_norm:.4e}, min={grad_min:.4e}, max={grad_max:.4e}, nan={has_nan}")
            
            if (torch.isinf(param.grad.norm())) or (torch.isinf(param.grad.min())) or (torch.isinf(param.grad.max())):
                inf_check = True

    return inf_check

def compute_drift_penalty(model, pretrained_weights, lambda_reg=0.001):
    '''
    Computes lambda regularization values for pretrained to finetuning predictions;
    model should be one of the encoders.
    '''

    # pretrained_weights = {name: param.clone().detach()
    #                  for name, param in model.named_parameters()}
    # print('model named params: ')
    # for name,_ in model.named_parameters():
    #     print('name: ', name)

    print('pretrained_weights keys: ')
    pretrained_weights_clone={}
    for name,val in pretrained_weights.items():
        pretrained_weights_clone['net_1d.'+name] = pretrained_weights[name]

    # for name in pretrained_weights_clone.keys():
    #     print('new name in pretrained: ', name)


    penalty = 0
    print_counter=0
    for name, param in model.named_parameters():
        # print(f'drift func param name: {name}')
        if name in pretrained_weights_clone:
            # print(f'param in model: std: {param.std().item()}, norm: {param.norm().item()}')
            # print(f'param in pretrained weights: std: {pretrained_weights_clone[name].std().item()}, norm: {pretrained_weights_clone[name].norm().item()}')
            # print('param: ', param)
            # print('pretrained_weights[name]: ', pretrained_weights_clone[name])
            penalty += torch.norm(param - pretrained_weights_clone[name])**2
            if print_counter in [0,5,15]:
                print(f'example of added penalty weight #{print_counter}: {(torch.norm(param - pretrained_weights_clone[name])**2)}')
            print_counter+=1

    return lambda_reg * penalty


def print_gradients(model, model_name="EGNN", norms=[], stds=[], grad_vals={}, epoch=0):
    inf_check=False
    print(f"\n--- Gradients for {model_name} ---")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_min = param.grad.min().item()
            grad_max = param.grad.max().item()
            grad_std = param.grad.std().item()
            has_nan = torch.isnan(param.grad).any().item()
            
            # log
            norms+=[grad_norm]
            stds+=[grad_std]
            if name not in grad_vals.keys():
                grad_vals[epoch][name]={'norm':[grad_norm],
                                    'min':[grad_min],
                                    'max':[grad_max],
                                    'std':[grad_std]}

            else:
                grad_vals[epoch][name]['norm']+=[grad_norm]
                grad_vals[epoch][name]['min']+=[grad_min]
                grad_vals[epoch][name]['max']+=[grad_max]
                grad_vals[epoch][name]['std']+=[grad_std]

            print(f"{name}: norm={grad_norm:.4e}, min={grad_min:.4e}, max={grad_max:.4e}, nan={has_nan}, std={grad_std:.8e}")
            
            if (torch.isinf(param.grad.norm())) or (torch.isinf(param.grad.min())) or (torch.isinf(param.grad.max())):
                inf_check=True
        else:
            print(f"{name}: No grad (frozen or unused)")
    return inf_check
        

def train_epoch(args, loader, epoch, model, model_dp, model_ema, ema, device, dtype, property_norms, optim,
                nodes_dist, gradnorm_queue, dataset_info, prop_dist, grad_scaler=None,
                loss_logging_dict={}, train_target_mad_mean=None, grad_vals={}, net_1d_pretrained_weights=None):
    '''
    Performs one training epoch;
     logs mean batch and mean epoch loss to wandb;
     prints out current batch stats every *args.n_report_steps* iterations;
     samples and visualizes some generated molecule chains every *args.visualize_every_batch* iterations;
    '''
    # print('pretrained weights: ', net_1d_pretrained_weights)

    model_dp.train() # set model and parallel-data model to train modes
    model.train()

    nll_epoch = [] # saving epochs' loss values in list (these are the total loss vals, so either combined loss or mse loss (downstream))
    loss_1d = [] # these are individual losses for pretraining
    loss_3d = []
    loss_contrastive = []

    norms=[]
    stds=[]

    prop_preds = []
    rmses = []
    r2s = []

    n_iterations = len(loader) # number of batches created from training dataset = nr of iterations in an epoch

    
    # here comes out the batch of our dataloder, i.e.: {'1d': {1d input (i.e. input_ids+attention_mask)}, '3d': {3d input}} --> adjust accordingly.
    for i, data in enumerate(loader): # take a batch for iteration i / n_iterations from the train dataloader
        batch_1d = data['1d'] # input_ids and attention_mask
        # batch_3d could have the field 'target', but only if finetuning!
        batch_3d = data['3d'] # what are we going to do with these input ids (e.g. dataset class & model forward)?, figure out how to add collatorforMLM to collate fn!(?)

        # 1d; check processing, data class/collate (in/out), forward/main func., also check todo list underneath polyMRL class.
        input_ids = batch_1d['input_ids'].to(device)
        attention_mask = batch_1d['attention_mask'].to(device)
        labels = batch_1d['labels'].to(device)

        # print('batch_1d: ', batch_1d)

        # 3d
        x = batch_3d['positions'].to(device, dtype) # features of these data batches are: positions (I think 3-dimensional matrices), atom_masks (dont know),
        node_mask = batch_3d['atom_mask'].to(device, dtype).unsqueeze(2) # as well as edge_masks (dont know), charges (1-dimensional node features I think), 
        edge_mask = batch_3d['edge_mask'].to(device, dtype) # and something called one-hot (dont know yet what feature this is)
        one_hot = batch_3d['one_hot'].to(device, dtype)
        charges = (batch_3d['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype) # these are not necessarily present,

        if args.train_mode=='property_prediction':
            targets = batch_3d['target'].to(device, dtype)

        print('data in train_test: ')
        # print('x (mean not removed): ', x[0:10,:,:], x.shape)
        # print('node_mask: ', node_mask.shape)
        # print('edge_mask: ', edge_mask, edge_mask.shape)
        # print('one_hot: ', one_hot.shape)
        # print('charges: ', charges.shape)
        # but are present in the QM9 dataset and can be included using the charges arg param

        # this centers positions to the coordinate center (0,0,0) (we remove the mean of every molecule's coordinates from every position vector)
        # print_diagnostics('x_before_mean', x)
        # print('x_before_mean:', x[0:5,:,:],x.shape)

        if n_iterations>=35:
            iter_limit=35
        else:
            iter_limit=n_iterations-1
        if args.unfreeze_encoders and epoch==0 and i==iter_limit:
            print('before unfreezing')
            for name, param in model.named_parameters():
                print(f'name: {name}, requires_grad: {param.requires_grad}')
            print('unfreezing encoders..')
            model.unfreeze_encoders_func()
            print('after unfreezing')
            for name, param in model.named_parameters():
                print(f'name: {name}, requires_grad: {param.requires_grad}')
        if args.unfreeze_projections and epoch==0 and i==iter_limit:
            model.unfreeze_projections_func()
            print('unfreezing projection layers..')
        
        x = remove_mean_with_mask(x, node_mask)
        # print('x mean removed: ', x[0:5,:,:], x.shape)
        # print_diagnostics('x_mean_removed', x)


        # note: once we know what the node mask is --> check out what the remove_mean function does
        
        # adding augmentation noise and/or rotations to the coordinate positions/x
        if args.augment_noise > 0: # in case we want to augment with noise I think?
            # Add noise eps ~ N(0, augment_noise) around points.
            eps = sample_center_gravity_zero_gaussian_with_mask(x.sifze(), x.device, node_mask)
            # print('eps: ', eps.shape)

            x = x + eps * args.augment_noise # some small eps (determined how) multiplied with this augment noise is added as noise to x

        # center it again, now with potentially augm. noise added; we need to do this because if we would not do it the first time, adding zero-center gaussian noise would go wrong (I think)
        x = remove_mean_with_mask(x, node_mask) # why two times?? This overwrites the augmentation noise addition from above
        # print('x mean removed again: ', x[0:10,:,:], x.shape)
        # print_diagnostics('x_mean_removed_2', x)

        if args.data_augmentation:
            x = utils.random_rotation(x).detach() # if augment is desired, add augmented data, namely rotated molecules

        # bs, atom_nr = charges.shape[0], charges.shape[1]
        
        # for b in range(bs):
        #     for at_nr, at in enumerate(range(atom_nr)):
        #         ch = charges[b][at]
        #         m = node_mask[b][at][0] # has 1-dim. at -1
        #         oh = one_hot[b][at]
        #         xx = x[b][at]

        #         print(f'molecule {b}, atom {at}: ', ch, m, oh, xx)
        #         if ch!=0 and m==0:
        #             print(f'atom {at_nr}: {at} wrong')
        #             print(charges[b], node_mask[b])
        #         elif ch==0 and m!=0:
        #             print(f'atom {at_nr}: {at} wrong')
        #             print(charges[b], node_mask[b])
        #         for ohe in oh:
        #             if ohe!=0 and m==0:
        #                 print(f'atom {at_nr}: {at} wrong')
        #                 print(one_hot[b][at_nr], node_mask[b][at_nr])
        #             elif ohe==0 and m!=0:
        #                 print(f'atom {at_nr}: {at} wrong')
        #                 print(one_hot[b][at_nr], node_mask[b][at_nr])

        #         for xxx in xx:
        #             if xxx!=0 and m==0:
        #                 print(f'atom {at_nr}: {at} wrong')
        #                 print(x[b][at_nr], node_mask[b][at_nr])
        #             elif xxx==0 and m!=0:
        #                 print(f'atom {at_nr}: {at} wrong')
        #                 print(x[b][at_nr], node_mask[b][at_nr])

        # for i,mask in enumerate(node_mask):
            
        #     print(f'mask: {i}', mask)
        # for i,ch in enumerate(charges):
        #     print(f'charges: {i}', ch)   

        check_mask_correct([x, one_hot, charges], node_mask) # check what the data holds
        
        # check out this function! --> checks if masks are applied correctly; what this means intrisically/shape wise, not so sure

        assert_mean_zero_with_mask(x, node_mask) # (this one too)


        # define h vector, which is (I think) the node data matrix/vector
        h = {'categorical': one_hot, 'integer': charges} # why are charges listed as integer and not floats? is one_hot maybe the atom type vector?
        print('h shape: ', h['categorical'].shape, h['integer'].shape) # (batch, seq_len, 5(=len(vocab))), (batch, seq_len, 1(=integer of atom nrs.))
        # now we have h and x!

        # note: can leave this prepare context and definition of the context variable in for now, not enabling conditioning option right now anyway
        # in case we condition on a molecular property, we turn that property into a tensor named 'context' here
        if len(args.conditioning) > 0:
            context = qm9utils.prepare_context(args.conditioning, batch_3d, property_norms).to(device, dtype) # molecule property = the context I think
            print('context: ', context.shape)
            assert_correctly_masked(context, node_mask) # check this out
        else:
            context = None


        # transform batch through model-flow
        # from 'compute_loss_and_nll.py'
        bs, n_nodes, n_dims = x.size()
        # edge_mask = edge_mask.view(bs, n_nodes * n_nodes) # -> included in forward!
        # assert_correctly_masked(x, node_mask) # -> included in forward!


        batch_1d_3d = {'1d': {'input_ids': input_ids, 'attention_mask': attention_mask,
                              'labels': labels},
                       '3d': {'x': x, 'h': h, 'node_mask': node_mask,
                            'edge_mask': edge_mask, 'context': context}
                            }
        # if we have downstream property prediction targets, add them to the 3d batch
        if args.train_mode=='property_prediction':
            batch_1d_3d['3d']['target'] = targets

            # improvised, but add mad and mean to the batch itself for target normalization
            batch_1d_3d['3d']['mad'] = train_target_mad_mean['mad']
            batch_1d_3d['3d']['mean'] = train_target_mad_mean['mean']

        device_type = "cuda" if args.cuda else "cpu"

        assert args.amp_dtype in ['16', '32']
        amp_dtype = torch.float16 if args.amp_dtype=='16' else torch.float32
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=args.mixed_precision_training):
            loss_1d_3d_contr, logs, net_1d_params = model(batch_1d_3d) # nll in EDM impl.


        # print('---printing model grad settings: ---')
        # for (n,param) in model.named_parameters():
        #     if 'net_1d' in n:
        #         print(f'parameter {n}: std: {param.std()}, norm: {param.norm()}')

        # print('---printing 1d encoder grad settings: ---')
        # for (n,param) in net_1d_params:
        #     print(f'parameter {n}: std: {param.std()}, norm: {param.norm()}')
            
        # note: not for now log_pn
        # N = node_mask.squeeze(2).sum(1).long()
        # log_pN = nodes_dist.log_prob(N)
        # assert nll.size() == log_pN.size()
        # nll = nll - log_pN
        # nll = nll.mean(0)

        # note: not for now regularization term
        # reg_term = torch.tensor([0.]).to(nll.device)
        # mean_abs_z = 0.

        # define loss as the neg. log likelihood output
        # loss = nll # + args.ode_regularization * reg_term # loss calc
        

        # (old)
        # print('x going into compute_loss_and_nll: ', x, x.shape)
        # nll, reg_term, mean_abs_z = losses.compute_loss_and_nll(args, model_dp, nodes_dist, # function to compute loss value,
                                                                # x, h, node_mask, edge_mask, context) # gives us regularization term and some type of value for z too
        # print('nll: ', nll)
        # print('reg_term: ', reg_term)
        # print('mean_abs_z: ', mean_abs_z)
        # standard nll from forward KL; the loss computed below is the one the model is trained on/gradient is computed based on
        # loss = nll + args.ode_regularization * reg_term # loss calc

        loss = loss_1d_3d_contr
        norm_penalty = compute_drift_penalty(model, net_1d_pretrained_weights)
        print('norm_penalty:', norm_penalty)
        print('loss_1d_3d_contr; loss: ', loss)
        loss += norm_penalty

        # gradient-scaling version of backward gradient computation
        grad_scaler.scale(loss).backward() # back propagation deriv
        # default version of backward
        # loss.backward() # back propagation deriv

        infs_present = check_infs(model, model_name=args.exp_name)

        if i%30==0 or infs_present:
            print("Inf detected-printing gradients" if infs_present else f"Iteration {i}-printing gradients")
            inf_check = print_gradients(model, model_name=args.exp_name, norms=norms, stds=stds, grad_vals=grad_vals, epoch=epoch)
            if inf_check:
                print('input_ids: ', input_ids)
                print('attention_mask: ', attention_mask)
                print('labels: ', labels)
                print('x: ', x)
                print('node_mask: ', node_mask)
                print('edge_mask: ', edge_mask)
                print('one_hot: ', one_hot)
                print('charges: ', charges)


        print('scale 1: ', grad_scaler.get_scale())
        if args.unscale:
            grad_scaler.unscale_(optim)
        print('scale 2: ', grad_scaler.get_scale())

        # calculate and define amount of gradient after gradient clipping, in case gradient clipping is desired, else just 0
        if args.clip_grad:
            grad_norm = utils.gradient_clipping(model, gradnorm_queue)
        else:
            grad_norm = 0.

        # gradient-scaling version of gradient step
        grad_scaler.step(optim) # grad descent step
        grad_scaler.update()

        # default version of gradient update step
        # optim.step() # grad descent step


        # Update EMA if enabled.
        if args.ema_decay > 0:
            ema.update_model_average(model_ema, model)
        
        optim.zero_grad()


        if i % args.n_report_steps == 0: # record epoch, iteration, and loss terms after x amount of steps determined by a pre-set arg
            print(f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, "
                  f"Loss {loss.item():.2f}, NLL: {loss_1d_3d_contr.item():.2f}, "
                  f"RegTerm: {0}, "
                  f"GradNorm: {grad_norm:.1f}")

        # note: so we log every batch's total/final loss value in the epoch, and the average of all the batches' loss values, i.e. one mean loss for the entire epoch;
        nll_epoch.append(loss_1d_3d_contr.item())
        if args.train_mode=='pretrain':
            loss_1d.append(logs['loss_1d'])
            loss_3d.append(logs['loss_3d'])
            loss_contrastive.append(logs['loss_contrastive'])
        elif args.train_mode=='property_prediction':
            prop_preds.append(logs['property_predictions'])
            rmses.append(logs['rmses'])
            r2s.append(logs['r2s'])

        if (epoch % args.test_epochs == 0) and (i % args.visualize_every_batch == 0) and not (epoch == 0 and i == 0): # if we are on a batch vis and test epoch iteration/epoch,
            start = time.time()                                                # and not on iteration 0 of epoch 0, then we

            ## note: commented out until analysing/testing func. exists.
            # so in every epoch, molecules are generated ('sampled'), and saved to the device.
            # also visualizations of (some of) these generated molecules are saved to the device.
            # -> note: we use the ema-model version also for sampling/generation of molecules.

            # if len(args.conditioning) > 0:                                     # sample chains and save them to the device
            #     save_and_sample_conditional(args, device, model_ema, prop_dist, dataset_info, epoch=epoch) # conditional sampling
            # save_and_sample_chain(model_ema, args, device, dataset_info, prop_dist, epoch=epoch, # 'normal' sampling
            #                       batch_id=str(i))
            # sample_different_sizes_and_save(model_ema, nodes_dist, args, device, dataset_info, # not sure what different size sampling means
            #                                 prop_dist, epoch=epoch)
                                            
            # print(f'Sampling took {time.time() - start:.2f} seconds')

            # # save visualizations of sampled (atom) chains / molecules
            # vis.visualize(f"outputs/{args.exp_name}/epoch_{epoch}_{i}", dataset_info=dataset_info, wandb=wandb)
            # vis.visualize_chain(f"outputs/{args.exp_name}/epoch_{epoch}_{i}/chain/", dataset_info, wandb=wandb)
            # if len(args.conditioning) > 0:
            #     vis.visualize_chain("outputs/%s/epoch_%d/conditional/" % (args.exp_name, epoch), dataset_info,
            #                         wandb=wandb, mode='conditional')

        wandb.log({"Batch NLL": loss_1d_3d_contr.item()}, commit=True) # log loss of the current batch to wandb

        if args.break_train_epoch: # not sure what this means; maybe if we only want to run model training for 1 epoch?
            break
    wandb.log({"Train Epoch NLL": np.mean(nll_epoch)}, commit=False) # logging the final, average loss over the entire epoch to wandb
    
    # log gradient info
    gradient_info = {}
    gradient_info['norms'] = norms
    gradient_info['stds'] = stds

    # with open(f'gradient_vals_{args.exp_name}_epoch_{epoch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json', 'w', encoding='utf-8') as f:
    #     json.dump(gradient_info, f, ensure_ascii=False, indent=4)

    if 'Train' in loss_logging_dict.keys():
        loss_logging_dict['Train'][epoch] = nll_epoch

        loss_logging_dict['Train_1d'][epoch] = loss_1d
        loss_logging_dict['Train_3d'][epoch] = loss_3d
        loss_logging_dict['Train_contrastive'][epoch] = loss_contrastive

        loss_logging_dict['Property_predictions'][epoch] = prop_preds
        loss_logging_dict['RMSE'][epoch] = rmses
        loss_logging_dict['R2'][epoch] = r2s

    else:
        loss_logging_dict['Train'] = {}

        loss_logging_dict['Train_1d'] = {}
        loss_logging_dict['Train_3d'] = {}
        loss_logging_dict['Train_contrastive'] = {}
        loss_logging_dict['Property_predictions'] = {}
        loss_logging_dict['RMSE'] = {}
        loss_logging_dict['R2'] = {}


        loss_logging_dict['Train'][epoch] = nll_epoch

        loss_logging_dict['Train_1d'][epoch] = loss_1d
        loss_logging_dict['Train_3d'][epoch] = loss_3d
        loss_logging_dict['Train_contrastive'][epoch] = loss_contrastive
        loss_logging_dict['Property_predictions'][epoch] = prop_preds
        loss_logging_dict['RMSE'][epoch] = rmses
        loss_logging_dict['R2'][epoch] = r2s


def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def test(args, loader, epoch, eval_model, device, dtype, property_norms, nodes_dist, partition='Test', loss_logging_dict={}, train_target_mad_mean=None):
    '''Perform one test epoch, over entire test loader/test set; 
        returns average test set loss, and prints out current average test loss per args.n_report_steps iterations'''

    # set model to evaluation mode and set torch.no_grad() (no gradient updates desired)
    eval_model.eval()
    with torch.no_grad():

        # initialize validation/testing epoch loss and number of samples; the total nr. of molecules fed in this epoch (can be different from num. iters. bc of unbalanced batches)
        nll_epoch = 0
        n_samples = 0
        
        nlls_epoch_all = []
        loss_1d = [] # these are individual losses for pretraining
        loss_3d = []
        loss_contrastive = []

        prop_preds = []
        rmses = []
        r2s = []

        n_iterations = len(loader) # number of batches? Not number of data points right?
        print('n_iterations: ', n_iterations)

        nlls_epoch_rolling = []
        
        for i, data in enumerate(loader): # get the next batch from the test dataloader
            batch_1d = data['1d'] # input_ids and attention_mask
            # batch_3d could have the field 'target', but only if finetuning!
            batch_3d = data['3d']
            # print('batch 3d valid: ', batch_3d['positions'])

            input_ids = batch_1d['input_ids'].to(device)
            attention_mask = batch_1d['attention_mask'].to(device)
            labels = batch_1d['labels'].to(device)

            # 3d
            x = batch_3d['positions'].to(device, dtype) # features of these data batches are: positions (I think 3-dimensional matrices), atom_masks (dont know),
            batch_size = x.size(0)
            node_mask = batch_3d['atom_mask'].to(device, dtype).unsqueeze(2) # as well as edge_masks (dont know), charges (1-dimensional node features I think)
            edge_mask = batch_3d['edge_mask'].to(device, dtype) # and something called one-hot (dont know yet what feature this is); # what is the one_hot feature?
            one_hot = batch_3d['one_hot'].to(device, dtype)
            charges = (batch_3d['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype) # get atom charge matrix if desired
                         

            
            if args.train_mode=='property_prediction':
                targets = batch_3d['target'].to(device, dtype)

            # print('batch 3d: ', batch_3d)

            x = remove_mean_with_mask(x, node_mask)

            if args.augment_noise > 0:
                # Add noise eps ~ N(0, augment_noise) around points.
                eps = sample_center_gravity_zero_gaussian_with_mask(x.size(), x.device, node_mask)
                x = x + eps * args.augment_noise # add noise augmentations in case we pre-select that we want noise augments in the args


            # center it again, now with potentially augm. noise added; we need to do this because if we would not do it the first time, adding zero-center gaussian noise would go wrong (I think)
            x = remove_mean_with_mask(x, node_mask)


            check_mask_correct([x, one_hot, charges], node_mask) # check if masks are applied correctly; what this means intrisically/shape wise, not so sure
            assert_mean_zero_with_mask(x, node_mask) # (this one too)

            # define h vector, which is (I think) the node data matrix/vector
            h = {'categorical': one_hot, 'integer': charges} # now we have x and h vectors/matrices, ready to put as input to model


            if len(args.conditioning) > 0:
                context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype) # get conditioned/property context if desired

                assert_correctly_masked(context, node_mask) # not sure what this does
            else:
                context = None

            batch_1d_3d = {'1d': {'input_ids': input_ids, 'attention_mask': attention_mask,
                              'labels': labels},
                       '3d': {'x': x, 'h': h, 'node_mask': node_mask,
                            'edge_mask': edge_mask, 'context': context}
                            }

            # if we have downstream property prediction targets, add them to the 3d batch
            if args.train_mode=='property_prediction':
                batch_1d_3d['3d']['target'] = targets

                # improvised, but add mad and mean to the batch itself for target normalization
                batch_1d_3d['3d']['mad'] = train_target_mad_mean['mad']
                batch_1d_3d['3d']['mean'] = train_target_mad_mean['mean']

            device_type = "cuda" if args.cuda else "cpu"
            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=args.mixed_precision_training):
                loss_1d_3d_contr, logs, _ = eval_model(batch_1d_3d) # nll in EDM impl.


            # transform batch through flow --> from this function we get the current batch's nll loss value
            # nll, _, _ = losses.compute_loss_and_nll(args, eval_model, nodes_dist, x, h, # I think this function computes the loss for a batch
            #                                         node_mask, edge_mask, context) # by running the forward on x and h, and computing the loss subsequently
            # --> runnig the forward here means taking a noise sample and denoising it and seeing what kind of molecule comes out (right?)                                                    
            # standard nll from forward KL
            
            nll = loss_1d_3d_contr
            # print('nll-iteration: ', nll)


            # take mean batch loss, multiply it with batch size to get full batch's loss,
            # and add it to the float nll_epoch, to get one value for all the losses in this valid./test epoch;
            nll_epoch += nll.item() * batch_size # fill in nll loss value #batch_size times into the loss value list (right?)
            # print('nll item, batch size: ', nll.item(), batch_size)
            # print('nll-epoch:',  nll_epoch)

            nlls_epoch_all.append(nll.item())
            if 'loss_1d' in logs.keys():
                loss_1d.append(logs['loss_1d']) # these are individual losses for pretraining
            if 'loss_3d' in logs.keys():
                loss_3d.append(logs['loss_3d'])
            if 'loss_contrastive' in logs.keys():
                loss_contrastive.append(logs['loss_contrastive'])
            if 'property_predictions' in logs.keys():
                prop_preds.append(logs['property_predictions'])
                rmses.append(logs['rmses'])
                r2s.append(logs['r2s'])

            # add nr. of mols in batch to total nr. of samples
            n_samples += batch_size # what about the last batch? Maybe that one is not as big as the given batch_size..? (in that case we should not add batch_size to n_samples in the last iteration I guess)
            # print('nr samples:', n_samples)

            # now, below we divide the total nll_epoch loss by the total nr. samples to get the validation model's loss value per molecule
            if i % args.n_report_steps == 0: # report_steps is the desired logging frequency, if at this frequency at batch i, we log the current loss values
                print(f"\r {partition} NLL \t epoch: {epoch}, iter: {i}/{n_iterations}, "
                      f"NLL: {nll_epoch/n_samples:.2f}") # log, i.e. print out, current mean nll over the test set (i.e. epoch) thus far
                                                         # note: it is common practice to not log individual batch val losses, but rather the epoch val loss,
                                                         # or the cumulative and eventually mean epoch val loss, as we do here.
            
            # log the current mean epoch nll/loss;
            nlls_epoch_rolling.append(nll_epoch/n_samples)

        if partition in loss_logging_dict.keys():
            loss_logging_dict[partition][epoch] = nlls_epoch_all
            loss_logging_dict[partition+'_1d'][epoch] = loss_1d
            loss_logging_dict[partition+'_3d'][epoch] = loss_3d
            loss_logging_dict[partition+'_contrastive'][epoch] = loss_contrastive
            loss_logging_dict[partition+'_property_predictions'][epoch] = prop_preds
            loss_logging_dict[partition+'_rmses'][epoch] = rmses
            loss_logging_dict[partition+'_r2s'][epoch] = r2s

            loss_logging_dict[partition+'_rolling'][epoch] = nlls_epoch_rolling
            
        else:
            loss_logging_dict[partition] = {}
            loss_logging_dict[partition+'_1d'] = {}
            loss_logging_dict[partition+'_3d'] = {}
            loss_logging_dict[partition+'_contrastive'] = {}
            loss_logging_dict[partition+'_rolling'] = {}
            loss_logging_dict[partition+'_property_predictions'] = {}
            loss_logging_dict[partition+'_rmses'] = {}
            loss_logging_dict[partition+'_r2s'] = {}

            loss_logging_dict[partition][epoch] = nlls_epoch_all
            loss_logging_dict[partition+'_1d'][epoch] = loss_1d
            loss_logging_dict[partition+'_3d'][epoch] = loss_3d
            loss_logging_dict[partition+'_contrastive'][epoch] = loss_contrastive
            loss_logging_dict[partition+'_rolling'][epoch] = nlls_epoch_rolling
            loss_logging_dict[partition+'_property_predictions'][epoch] = prop_preds
            loss_logging_dict[partition+'_rmses'][epoch] = rmses
            loss_logging_dict[partition+'_r2s'][epoch] = r2s


    # return the average loss value per molecule, over all the samples in this valid/test epoch's data samples
    return nll_epoch/n_samples


def save_and_sample_chain(model, args, device, dataset_info, prop_dist,
                          epoch=0, id_from=0, batch_id=''):
    '''
    Extracts samples from the learned training data distribution, and gets molecule features out of them;
     then saves these to a file on device, and returns the one_hot, atom charge and coordinate matrices.
     
    '''

    one_hot, charges, x = sample_chain(args=args, device=device, flow=model, # is 'flow' maybe intended to be the reverse denoising distribution p(x-1|x)?
                                       n_tries=1, dataset_info=dataset_info, prop_dist=prop_dist)

    vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/chain/',
                      one_hot, charges, x, dataset_info, id_from, name='chain')

    return one_hot, charges, x


def sample_different_sizes_and_save(model, nodes_dist, args, device, dataset_info, prop_dist,
                                    n_samples=5, epoch=0, batch_size=100, batch_id=''):
    '''
    Sample molecules of varying sizes (i.e. nr of atoms?) (or only 1 molecule?), and save them to device
    '''

    batch_size = min(batch_size, n_samples) # batch_size becomes the number of samples if the nr of samples is smaller than batch_size

    for counter in range(int(n_samples/batch_size)): # loop over every batch
        nodesxsample = nodes_dist.sample(batch_size) # sample x coordinates from learned node distribution

        # get sample molecules' features by sampling from the learned distribution, given (property distribution and) x samples
        one_hot, charges, x, node_mask = sample(args, device, model, prop_dist=prop_dist,
                                                nodesxsample=nodesxsample,
                                                dataset_info=dataset_info)

        # print out the coordinates ('x') of the generated molecule, and save the molecule features onto device
        print(f"Generated molecule: Positions {x[:-1, :, :]}")
        vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/', one_hot, charges, x, dataset_info,
                          batch_size * counter, name='molecule')


def analyze_and_save(epoch, model_sample, nodes_dist, args, device, dataset_info, prop_dist,
                     n_samples=1000, batch_size=100):
    '''
    function for calculating the stability, and other metrics, of a set of new, sampled molecules,
            that we are able to generate after training the models,
            i.e. samples coming out of our model's learned data distribution;

    input: sampled molecules from trained model; the current epoch (so this analysis is done during training?);  data distribution of 
            node data and property data (if conditonal?); number of samples made; pre-set batch size used for the model's forward;

    output: dictionary with stability value for every sample molecule that we get from our sampling action;
             additionally all calculated metrics (stability, validity, uniqueness, novelty), are logged to the current wandb exp.
    '''

    print(f'Analyzing molecule stability at epoch {epoch}...')

    batch_size = min(batch_size, n_samples) # if less samples than one batch, the batch size becomes the total nr of samples
    assert n_samples % batch_size == 0 # all batches must be of equal size, e.g. no smaller last batch

    molecules = {'one_hot': [], 'x': [], 'node_mask': []} # initialize molecule samples dictionary

    for i in range(int(n_samples/batch_size)): # loop over all of our batches
        nodesxsample = nodes_dist.sample(batch_size) # sample node data according to our current (learned?) node distribution
        one_hot, charges, x, node_mask = sample(args, device, model_sample, dataset_info, prop_dist,
                                                nodesxsample=nodesxsample) # rest of samples we get from this sample function; one_hot, charges and the node mask (which is ...?)

        molecules['one_hot'].append(one_hot.detach().cpu())
        molecules['x'].append(x.detach().cpu())
        molecules['node_mask'].append(node_mask.detach().cpu()) # fill up molecule dict with our sampled data

    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
    validity_dict, rdkit_tuple = analyze_stability_for_molecules(molecules, dataset_info) # get stability values for all sampled molecules

    wandb.log(validity_dict) # log the stability values of the sample molecules to wandb
    if rdkit_tuple is not None: # in case we get an RDKit tuple out of the previous fnc
        # log the RDKit metrics on the sample molecules to wandb
        wandb.log({'Validity': rdkit_tuple[0][0], 'Uniqueness': rdkit_tuple[0][1], 'Novelty': rdkit_tuple[0][2]}) 
    return validity_dict # return the stability values


def save_and_sample_conditional(args, device, model, prop_dist, dataset_info, epoch=0, id_from=0):
    '''
    samples, given a property disribution, molecules from the a learned distribution that are steered towards some target property,
     and saves these molecules (all their features) to the device;
    
    output: the one_hot feature vector/matrix, as well as the atom charge and coordinate matrices.
    '''

    one_hot, charges, x, node_mask = sample_sweep_conditional(args, device, model, dataset_info, prop_dist)

    vis.save_xyz_file(
        'outputs/%s/epoch_%d/conditional/' % (args.exp_name, epoch), one_hot, charges, x, dataset_info,
        id_from, name='conditional', node_mask=node_mask)

    return one_hot, charges, x
