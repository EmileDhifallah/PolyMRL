# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import random
import re
import copy
import seaborn as sns
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
import torch
from torch.cuda.amp import autocast, GradScaler

import time
import pickle
import numpy as np
from pi1m.utils import prepare_context, compute_mean_mad
from pi1m.models import DistributionProperty, DistributionNodes
from train_test import train_epoch, test, analyze_and_save
from model.model_classes import polyMRL
import json
import matplotlib.pyplot as plt
import os
import pandas as pd

parser = argparse.ArgumentParser(description='E3Diffusion')
parser.add_argument('--exp_name', type=str, default='debug_10')

args = parser.parse_args()

print("args list: ", args)



def plot_training_logs(
    file_path='file.json',
    colors=('tab:blue', 'tab:orange', 'tab:green'),
    labels=('Train', 'Val', 'Test'),
    title='Training/Validation/Test Loss Over Iterations'
    ):
    """
    Plots training, validation, and test logs from a JSON file.

    Parameters:
        file_path (str): Path to the JSON file.
        colors (tuple): Tuple of 3 color strings for Train, Val, and Test.
        labels (tuple): Tuple of 3 label names for the legend.
        title (str): Title of the plot.
    """
    assert os.path.exists(file_path), f"File not found: {file_path}"

    with open(file_path, 'r') as f:
        data = json.load(f)

    # Collect Train data
    train_values = []
    for epoch in sorted(data["Train"].keys(), key=int):
        train_values.extend(data["Train"][epoch])

    # Collect Val and Test data (second value in each list assumed to be the relevant one)
    val_values = [v[1] for _, v in sorted(data["Val"].items(), key=lambda x: int(x[0]))]
    test_values = [v[1] for _, v in sorted(data["Test"].items(), key=lambda x: int(x[0]))]

    # Create x-axis values
    train_x = list(range(len(train_values)))
    val_x = list(range(len(train_values) - len(val_values), len(train_values)))
    test_x = list(range(len(train_values) - len(test_values), len(train_values)))

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(train_x, train_values, label=labels[0], color=colors[0])
    plt.plot(val_x, val_values, label=labels[1], color=colors[1], linestyle='--')
    plt.plot(test_x, test_values, label=labels[2], color=colors[2], linestyle=':')

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"plot_plt_{file_path.split('.')[0]}", dpi=300, bbox_inches='tight')
    plt.clf()

def plot_prediction_logs_no_train(
    file_path='file.json',
    labels=('Prediction-variability', 'Val', 'Test', 'Test-RMSE'),
    title="Standard deviation per iteration's regression predictions, together with validation/test losses",
    palette='colorblind',
    save_path='loss_plot.png',
    drop_iters=False
    ):
    assert os.path.exists(file_path), f"File not found: {file_path}"

    # Load JSON
    with open(file_path, 'r') as f:
        data = json.load(f)

    # ---- Collect all data ----
    records = []
    total_iter = 0

    for epoch in sorted(data["Val"].keys(), key=int): 
        epoch_predictions = data["Property_predictions"][epoch] 
        epoch_val = data["Val"][epoch]  
        epoch_test = data["Test"][epoch]
        epoch_rmse = data["Test_rmses"][epoch]

        # optional downsampling
        if drop_iters:
            keeping_idxs = np.linspace(0,len(epoch_predictions)-1,num=150)
            epoch_predictions = np.array(epoch_predictions)[keeping_idxs.astype(int)].tolist()

            keeping_idxs = np.linspace(0,len(epoch_val)-1,num=45)
            epoch_val = np.array(epoch_val)[keeping_idxs.astype(int)].tolist()

            keeping_idxs = np.linspace(0,len(epoch_test)-1,num=45)
            epoch_test = np.array(epoch_test)[keeping_idxs.astype(int)].tolist()

            keeping_idxs = np.linspace(0,len(epoch_rmse)-1,num=45)
            epoch_rmse = np.array(epoch_rmse)[keeping_idxs.astype(int)].tolist()

        # repeat val/test/rmse to match predictions length
        len_pred = len(epoch_predictions)

        if len(epoch_val) < len_pred:
            len_val = len(epoch_val)
            epoch_val_new = (epoch_val * (len_pred // len_val))
            epoch_val_new += epoch_val[:(len_pred % len_val)]
            epoch_val = epoch_val_new
        
        if len(epoch_test) < len_pred:
            len_test = len(epoch_test)
            epoch_test_new = (epoch_test * (len_pred // len_test))
            epoch_test_new += epoch_test[:(len_pred % len_test)]
            epoch_test = epoch_test_new

        if len(epoch_rmse) < len_pred:
            len_rmse = len(epoch_rmse)
            epoch_rmse_new = (epoch_rmse * (len_pred // len_rmse))
            epoch_rmse_new += epoch_rmse[:(len_pred % len_rmse)]
            epoch_rmse = epoch_rmse_new

        n_iter = len(epoch_predictions)

        for i, (prediction_val, val_loss, test_loss, test_rmse) in enumerate(
            zip(epoch_predictions, epoch_val, epoch_test, epoch_rmse)
        ):
            iter_idx = total_iter + i
            records.append({'Iteration': iter_idx, 'batch std': prediction_val, 'Split': labels[0]})
            records.append({'Iteration': iter_idx, 'batch loss': val_loss,  'Split': labels[1]})
            records.append({'Iteration': iter_idx, 'batch loss': test_loss, 'Split': labels[2]})
            records.append({'Iteration': iter_idx, 'batch loss': test_rmse, 'Split': labels[3]})
        
        total_iter += n_iter

    # ---- Create dataframe ----
    df = pd.DataFrame(records)

    custom_palette = {
        labels[1]: '#999999',  
        labels[2]: '#383838',  
        labels[3]: '#6F9484',  
        labels[0]: '#6F94BD'   
    }

    custom_dashes = {
        labels[1]: (1, 1),     
        labels[2]: (4, 1),     
        labels[3]: (4, 1, 1),  
        labels[0]: ''          
    }

    # ---- Plot ----
    sns.set(style='whitegrid', palette=palette, font_scale=1., font="Serif")
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x='Iteration', y='batch std', hue='Split', palette=custom_palette, linewidth=2.2, style='Split', dashes=custom_dashes, legend=False)
    
    ax2 = plt.twinx()
    sns.lineplot(data=df, x='Iteration', y='batch loss', hue='Split', palette=custom_palette, linewidth=1.4, style='Split', dashes=custom_dashes, ax=ax2)

    y_lim = max(epoch_predictions + epoch_val + epoch_test + epoch_rmse) + 3
    ax2.set_ylim(bottom=0, top=y_lim)
    plt.title(title)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_prediction_logs_no_train_no_preds(
    file_path='file.json',
    labels=('Train','Val', 'Test'),
    title="Train, validation and test loss on the ",
    palette='colorblind',
    save_path='loss_plot.png',
    drop_iters=False
    ):
    assert os.path.exists(file_path), f"File not found: {file_path}"

    # Load JSON
    with open(file_path, 'r') as f:
        data = json.load(f)

    # ---- Collect all data ----
    records = []
    total_iter = 0

    for epoch in sorted(data["Val"].keys(), key=int): 
        epoch_train = data[labels[0]][epoch]
        epoch_val = data[labels[1]][epoch]  
        epoch_test = data[labels[2]][epoch]
        

        # optional downsampling
        if drop_iters:
            
            keeping_idxs = np.linspace(0,len(epoch_train)-1,num=45)
            epoch_train = np.array(epoch_train)[keeping_idxs.astype(int)].tolist()

            keeping_idxs = np.linspace(0,len(epoch_val)-1,num=45)
            epoch_val = np.array(epoch_val)[keeping_idxs.astype(int)].tolist()

            keeping_idxs = np.linspace(0,len(epoch_test)-1,num=45)
            epoch_test = np.array(epoch_test)[keeping_idxs.astype(int)].tolist()


        # repeat val/test/rmse to match predictions length
        len_pred = len(epoch_train)
        
        if len(epoch_train) < len_pred:
            len_train = len(epoch_train)
            epoch_train_new = (epoch_train * (len_pred // len_train))
            epoch_train_new += epoch_train[:(len_pred % len_train)]
            epoch_train = epoch_train_new

        if len(epoch_val) < len_pred:
            len_val = len(epoch_val)
            epoch_val_new = (epoch_val * (len_pred // len_val))
            epoch_val_new += epoch_val[:(len_pred % len_val)]
            epoch_val = epoch_val_new
        
        if len(epoch_test) < len_pred:
            len_test = len(epoch_test)
            epoch_test_new = (epoch_test * (len_pred // len_test))
            epoch_test_new += epoch_test[:(len_pred % len_test)]
            epoch_test = epoch_test_new


        n_iter = len(epoch_train)

        for i, (train_loss, val_loss, test_loss) in enumerate(
            zip(epoch_train, epoch_val, epoch_test)
        ):
            iter_idx = total_iter + i
            records.append({'Iteration': iter_idx, 'batch std': train_loss, 'Split': labels[0]})
            records.append({'Iteration': iter_idx, 'batch loss': val_loss,  'Split': labels[1]})
            records.append({'Iteration': iter_idx, 'batch loss': test_loss, 'Split': labels[2]})
        
        total_iter += n_iter

    # ---- Create dataframe ----
    df = pd.DataFrame(records)

    custom_palette = {
        labels[1]: '#999999',  
        labels[2]: '#383838',  
        labels[0]: '#6F94BD'   
    }

    custom_dashes = {
        labels[1]: (1, 1),     
        labels[2]: (4, 1),     
        labels[0]: ''          
    }

    # ---- Plot ----
    sns.set(style='whitegrid', palette=palette, font_scale=1., font="Serif")
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x='Iteration', y='batch std', hue='Split', palette=custom_palette, linewidth=2.2, style='Split', dashes=custom_dashes, legend=False)
    
    ax2 = plt.twinx()
    sns.lineplot(data=df, x='Iteration', y='batch loss', hue='Split', palette=custom_palette, linewidth=1.4, style='Split', dashes=custom_dashes, ax=ax2)

    y_lim = max( epoch_val + epoch_test + epoch_train) + 3
    ax2.set_ylim(bottom=0, top=y_lim)
    plt.title(title)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')

def plot_prediction_logs(
    file_path='file.json',
    labels=('Prediction-variability', 'Train', 'Val', 'Test' ),
    title="Standard deviation per iteration's regression predictions, together with train/valid/test losses",
    palette='colorblind',
    save_path='loss_plot.png',
    drop_iters=False
    ):
    assert os.path.exists(file_path), f"File not found: {file_path}"

    # Load JSON
    with open(file_path, 'r') as f:
        data = json.load(f)

    # ---- Collect all data ----
    records = []
    total_iter = 0 # will go from 0 to nr_iters*nr_epochs, e.g. 30*11=330

    for epoch in sorted(data["Train"].keys(), key=int): # 0 to 29
        epoch_predictions = data["Property_predictions"][epoch] # [loss_iter_0, .., loss_iter_n]
        epoch_train = data["Train"][epoch] # [loss_iter_0, .., loss_iter_n]
        epoch_val = data["Val"][epoch]  # second value is the loss
        epoch_test = data["Test"][epoch]
        # print('train epoch: ', epoch_train)
        # use less iterations for less chaotic graph
        if drop_iters:

            keeping_idxs = np.linspace(0,len(epoch_train)-1,num=50)
            # print('keeping_idxs.astype(int)', keeping_idxs.astype(int))
            epoch_train = np.array(epoch_train)[keeping_idxs.astype(int)].tolist()

            keeping_idxs = np.linspace(0,len(epoch_predictions)-1,num=50)
            epoch_predictions = np.array(epoch_predictions)[keeping_idxs.astype(int)].tolist()

            keeping_idxs = np.linspace(0,len(epoch_val)-1,num=15)
            epoch_val = np.array(epoch_val)[keeping_idxs.astype(int)].tolist()

            keeping_idxs = np.linspace(0,len(epoch_test)-1,num=15)
            epoch_test = np.array(epoch_test)[keeping_idxs.astype(int)].tolist()

        # repeat val/test iter values to fill up arrays to the lenght of the train iter-array -> predictions array is as long as train_epoch array so we chilling
        len_train = len(epoch_train)
        if len(epoch_val)<len(epoch_train):
            len_val = len(epoch_val)    
            epoch_val_new = (epoch_val*(len_train//len_val))
            epoch_val_new += epoch_val[:(len_train%len_val)]
            epoch_val=epoch_val_new
        
        if len(epoch_test)<len(epoch_train):
            len_test = len(epoch_test)
            epoch_test_new = (epoch_test*(len_train//len_test))
            epoch_test_new += epoch_test[:(len_train%len_test)]
            epoch_test=epoch_test_new

        n_iter = len(epoch_train)

        for i, (prediction_val, train_loss,val_loss, test_loss) in enumerate(zip(epoch_predictions, epoch_train, epoch_val, epoch_test)):
            iter_idx = total_iter + i
            records.append({'Iteration': iter_idx, 'batch std': prediction_val, 'Split': labels[0]})
            records.append({'Iteration': iter_idx, 'batch MSE loss': train_loss, 'Split': labels[1]})
            records.append({'Iteration': iter_idx, 'batch MSE loss': val_loss,  'Split': labels[2]})
            records.append({'Iteration': iter_idx, 'batch MSE loss': test_loss, 'Split': labels[3]})
        
        total_iter += n_iter

    # ---- Create dataframe ----
    df = pd.DataFrame(records)

    custom_palette = {
    labels[1]: '#C4C4C4',  # black
    labels[2]: '#999999',  # dark gray
    labels[3]: '#383838',  # light gray
    labels[0]: '#6F94BD'   # red or any color you want to highlight
    }

    custom_dashes = {
    labels[1]: (4, 1, 1, 1),      # dashed
    labels[2]: (1, 1),      # dotted
    labels[3]: (4, 1), # dash-dot
    labels[0]: ''
    }

    # ---- Plot ----
    
    sns.set(style='whitegrid', palette=palette, font_scale=1., font="Serif")
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x='Iteration', y='batch std', hue='Split', palette=custom_palette, linewidth=2.2, style='Split', dashes=custom_dashes, legend=False)
    
    ax2 = plt.twinx()
    sns.lineplot(data=df, x='Iteration', y='batch MSE loss', hue='Split', palette=custom_palette, linewidth=1.4, style='Split', dashes=custom_dashes, ax=ax2)

    y_lim = max(epoch_predictions+epoch_train+epoch_val+epoch_test)+3
    ax2.set_ylim(bottom=0, top=y_lim)#, emit=True, auto=False, *, ymin=None, ymax=None)
    # plt.ylim(0, y_lim)  # <- Y-axis fixed scale, eg 5
    plt.title(title)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_logs_seaborn(
    file_path='file.json',
    labels=('Train', 'Val', 'Test'),
    title='Training, Validation, and Test Loss Over Iterations',
    palette='colorblind',
    save_path='loss_plot.png',
    drop_iters=False
    ):
    assert os.path.exists(file_path), f"File not found: {file_path}"

    # Load JSON
    with open(file_path, 'r') as f:
        data = json.load(f)

    # ---- Collect all data ----
    records = []
    total_iter = 0 # will go from 0 to nr_iters*nr_epochs, e.g. 30*11=330

    for epoch in sorted(data["Train"].keys(), key=int): # 0 to 29
        epoch_train = data["Train"][epoch] # [loss_iter_0, .., loss_iter_n]
        epoch_val = data["Val"][epoch]  # second value is the loss
        epoch_test = data["Test"][epoch]
        print('train epoch: ', epoch_train)
        if drop_iters:

            keeping_idxs = np.linspace(0,len(epoch_train)-1,num=50)
            # print('keeping_idxs.astype(int)', keeping_idxs.astype(int))
            epoch_train = np.array(epoch_train)[keeping_idxs.astype(int)].tolist()

            keeping_idxs = np.linspace(0,len(epoch_val)-1,num=15)
            epoch_val = np.array(epoch_val)[keeping_idxs.astype(int)].tolist()

            keeping_idxs = np.linspace(0,len(epoch_test)-1,num=15)
            epoch_test = np.array(epoch_test)[keeping_idxs.astype(int)].tolist()


        len_train = len(epoch_train)
        if len(epoch_val)<len(epoch_train):
            len_val = len(epoch_val)    
            epoch_val_new = (epoch_val*(len_train//len_val))
            epoch_val_new += epoch_val[:(len_train%len_val)]
            epoch_val=epoch_val_new
        
        if len(epoch_test)<len(epoch_train):
            len_test = len(epoch_test)
            epoch_test_new = (epoch_test*(len_train//len_test))
            epoch_test_new += epoch_test[:(len_train%len_test)]
            epoch_test=epoch_test_new

        n_iter = len(epoch_train)

        for i, (train_loss,val_loss, test_loss) in enumerate(zip(epoch_train, epoch_val, epoch_test)):
            iter_idx = total_iter + i
            records.append({'Iteration': iter_idx, 'Loss': train_loss, 'Split': labels[0]})
            records.append({'Iteration': iter_idx, 'Loss': val_loss,  'Split': labels[1]})
            records.append({'Iteration': iter_idx, 'Loss': test_loss, 'Split': labels[2]})
        
        total_iter += n_iter

    # ---- Create dataframe ----
    df = pd.DataFrame(records)

    # ---- Plot ----
    sns.set(style='whitegrid', palette=palette, font_scale=1.1)
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x='Iteration', y='Loss', hue='Split', linewidth=2.2)

    plt.ylim(0, 5)  # <- Y-axis fixed scale
    plt.title(title)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')



def plot_gradient_stats_over_epochs(
    file_list,
    kind='norms',                  # 'norms' or 'stds'
    save_path='grad_plot.png',
    title='Gradient Statistics Over Iterations',
    palette='colorblind'
):
    assert isinstance(file_list, list) and len(file_list) > 0, "file_list must be a non-empty list"

    raw_vals = []

    for file_path in file_list:
        assert os.path.exists(file_path), f"File not found: {file_path}"
        with open(file_path, 'r') as f:
            data = json.load(f)
            assert kind in data, f"'{kind}' not found in file: {file_path}"
            raw_vals.extend(data[kind])

    # Convert to numpy array for easier processing
    values = np.array(raw_vals, dtype=np.float64)

    # Compute max ignoring inf and nan
    finite_max = np.nanmax(values[np.isfinite(values)])

    # Replace inf/-inf with finite max, and nan with 0
    values[np.isposinf(values)] = finite_max
    values[np.isneginf(values)] = finite_max
    values[np.isnan(values)] = 0.0

    iterations = list(range(len(values)))
    y_max = np.max(values) + 10


    sns.set(style='whitegrid', palette=palette, font_scale=1.1)
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=iterations, y=values, linewidth=2.0)

    plt.ylim(0, y_max)
    plt.xlabel('Iteration')
    plt.ylabel('Gradient ' + ('Norm' if kind == 'norms' else 'Std'))
    plt.title(title)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')

# NOTE: comment out below part if importing this module from different script

if __name__ == "__main__":
    # gradient plotting
    file_list_old = ["gradient_info_final_model_1_epoch_0_20250727_153155.json",
            "gradient_info_final_model_1_epoch_1_20250727_170716.json"
            ]

    file_list = ["/home/14044994/test/Repos/e3_diffusion_for_molecules/gradient_vals_final_model_1d_QM9_epoch_0_20250728_162945.json",
            "/home/14044994/test/Repos/e3_diffusion_for_molecules/gradient_vals_final_model_1d_QM9_epoch_1_20250728_163915.json",
            "/home/14044994/test/Repos/e3_diffusion_for_molecules/gradient_vals_final_model_1d_QM9_epoch_2_20250728_164845.json",
            "/home/14044994/test/Repos/e3_diffusion_for_molecules/gradient_vals_final_model_1d_QM9_epoch_3_20250728_165815.json",
            "/home/14044994/test/Repos/e3_diffusion_for_molecules/gradient_vals_final_model_1d_QM9_epoch_4_20250728_170745.json",
            "/home/14044994/test/Repos/e3_diffusion_for_molecules/gradient_vals_final_model_1d_QM9_epoch_5_20250728_171715.json",
            "/home/14044994/test/Repos/e3_diffusion_for_molecules/gradient_vals_final_model_1d_QM9_epoch_6_20250728_172644.json",
            "/home/14044994/test/Repos/e3_diffusion_for_molecules/gradient_vals_final_model_1d_QM9_epoch_7_20250728_173614.json"
            ]

    # plot_gradient_stats_over_epochs(
    # file_list,
    # kind='stds',                  # 'norms' or 'stds'
    # save_path=f'plot_gradient_stds_{file_list[-1].split('/')[-1][:-5]}.png',
    # title='Gradient standard deviation values for all model parameters over pretraining iterations',
    # palette='muted'
    # )


    new_predictions = {"Eat":"iter_loss_dict_prop_pred_prediction_printing_Eat_20250731_035128.json",
        "Eea":"iter_loss_dict_prop_pred_prediction_printing_Eea_20250731_033444.json",
        "Egb":"iter_loss_dict_prop_pred_prediction_printing_Egb_20250731_033137.json",
        "Egc":"iter_loss_dict_prop_pred_prediction_printing_Egc_20250731_032739.json",
        "Ei":"iter_loss_dict_prop_pred_prediction_printing_Ei_20250731_033750.json",
        "EPS":"iter_loss_dict_prop_pred_prediction_printing_EPS_20250731_034507.json",
        "Nc":"iter_loss_dict_prop_pred_prediction_printing_Nc_20250731_034817.json",
        "Xc":"iter_loss_dict_prop_pred_prediction_printing_Xc_20250731_034154.json",
        }

    encoder_losses = {'1d':"iter_loss_dict_pretrain_1d_20250727_050407_epoch_1.json",
                            '3d':"iter_loss_dict_PI1M_mean_pool_3d_refl_true_diagn_new_norm_20250726_023118.json",
                            'combined':"iter_loss_dict_final_model_1_20250727_154901_epoch_0.json"
                            }

    old_predictions = {"Xc":"iter_loss_dict_prediction_3_final_model_1_Xc_20250727_203850_epoch_19",
                    "Nc":"iter_loss_dict_prediction_3_final_model_1_Nc_20250727_204401_epoch_13",
                    "EPS":"iter_loss_dict_prediction_3_final_model_1_EPS_20250727_204135_epoch_19",
                    "Ei":"iter_loss_dict_prediction_3_final_model_1_Ei_20250727_203549_epoch_19",
                    "Egc":"iter_loss_dict_prediction_3_final_model_1_Egc_20250727_202721_epoch_19",
                    "Egb":"iter_loss_dict_prediction_3_final_model_1_Egb_20250727_203030_epoch_19",
                    "Eea":"iter_loss_dict_prediction_3_final_model_1_Eea_20250727_203259_epoch_19",
                    "Eat":"iter_loss_dict_prediction_2_final_model_1_Eat_20250727_192054_epoch_9"
        }

    final_model_losses_loading_options = {"no-amp-fp32 loaded encoders": "iter_loss_dict_final_model_combined_PI1M_no-amp_loaded-PI1M-noampfp32-encoders_20250731_051538_epoch_3.json",
        }

    # finetuning loss plotting
    # for data_name, file in final_model_losses_loading_options.items():

    #     plot_logs_seaborn(
    #     file_path=f'{file}',
    #     labels=('Train', 'Val', 'Test'),
    #     title=f'{data_name}, loss values 4 epochs of pretraining; train, validation, and test loss shown',
    #     palette='muted',
    #     save_path=f'plot_sns_{file.split('.')[0]}.png',
    #     drop_iters=True
    #         )

        # plot_training_logs(
        # file_path=f'{file}.json',
        # colors=('tab:blue', 'tab:orange', 'tab:green'),
        # labels=('Train', 'Val', 'Test'),
        # title='Training/Validation/Test Loss Over Iterations'
        #     )


    prediction_grid_exps = {"Egb, Grid default setting": "output_files/property_prediction/loss_dicts/iter_loss_dict_prop_pred_prediction_printing_Egb-TESTING-GRID_20250804_185825.json",
                            "Egb, Freeze projections": "output_files/property_prediction/loss_dicts/iter_loss_dict_prop_pred_prediction_printing_Egb-TESTING-GRID_freeze-projections_20250804_190222.json",
                            "Egb, Variable LRs": "output_files/property_prediction/loss_dicts/iter_loss_dict_prop_pred_prediction_printing_Egb-TESTING-GRID_var-lrs_20250804_185951.json",
                            "Egc, Grid default setting": "output_files/property_prediction/loss_dicts/iter_loss_dict_prop_pred_prediction_printing_Egc-TESTING-GRID_20250804_185345.json",
                            "Egc, Freeze projections": "output_files/property_prediction/loss_dicts/iter_loss_dict_prop_pred_prediction_printing_Egc-TESTING-GRID_freeze-projections_20250804_185800.json",
                            "Egc, Variable LRs": "output_files/property_prediction/loss_dicts/iter_loss_dict_prop_pred_prediction_printing_Egc-TESTING-GRID_var-lrs_20250804_185527.json",
                            "Xc, Grid default setting":  "output_files/property_prediction/loss_dicts/iter_loss_dict_prop_pred_prediction_printing_Xc-TESTING-GRID_20250804_190249.json",
                            "Xc, Freeze projections":  "output_files/property_prediction/loss_dicts/iter_loss_dict_prop_pred_prediction_printing_Xc-TESTING-GRID_freeze-projections_20250804_190707.json",
                            "Xc, Variable LRs":  "output_files/property_prediction/loss_dicts/iter_loss_dict_prop_pred_prediction_printing_Xc-TESTING-GRID_var-lrs_20250804_190420.json",
        }
    
    prediction_grid_exps_2 = ["output_files/property_prediction/loss_dicts/iter_loss_dict_Egb-TESTING-GRID-batchsize8_20250806_044804.json",
                            "output_files/property_prediction/loss_dicts/iter_loss_dict_Egb-TESTING-GRID-freezeencoders_20250806_034853.json",
                            "output_files/property_prediction/loss_dicts/iter_loss_dict_Egb-TESTING-GRID-freezeprojs-not-freezeencoders_20250806_043246.json",
                            "output_files/property_prediction/loss_dicts/iter_loss_dict_Egb-TESTING-GRID-highlr_20250806_025121.json",
                            "output_files/property_prediction/loss_dicts/iter_loss_dict_Egb-TESTING-GRID-highwd_20250806_035206.json",
                            "output_files/property_prediction/loss_dicts/iter_loss_dict_Egb-TESTING-GRID-lowlr_20250806_041232.json",
                            "output_files/property_prediction/loss_dicts/iter_loss_dict_Egb-TESTING-GRID-only1d_20250806_031912.json",
                            "output_files/property_prediction/loss_dicts/iter_loss_dict_Egb-TESTING-GRID-only3d_20250806_034429.json",
                            "output_files/property_prediction/loss_dicts/iter_loss_dict_Egb-TESTING-GRID-outputscale10_20250806_031928.json",
                            "output_files/property_prediction/loss_dicts/iter_loss_dict_Egb-TESTING-GRID-scale12-epochs100_20250806_050707.json",
                            "output_files/property_prediction/loss_dicts/iter_loss_dict_Egb-TESTING-GRID-unfreezeencoders_20250806_041339.json",
                            "output_files/property_prediction/loss_dicts/iter_loss_dict_Egb-TESTING-GRID-unfreezeprojections_20250806_041711.json",
                            "output_files/property_prediction/loss_dicts/iter_loss_dict_Egb-TESTING-GRID-unfreezeprojections_20250806_042212.json",
                            "output_files/property_prediction/loss_dicts/iter_loss_dict_Egc-TESTING-GRID-batchsize8_20250806_044329.json",
                            "output_files/property_prediction/loss_dicts/iter_loss_dict_Egc-TESTING-GRID-freezeencoders_20250806_034545.json",
                            "output_files/property_prediction/loss_dicts/iter_loss_dict_Egc-TESTING-GRID-freezeprojs-not-freezeencoders_20250806_042934.json",
                            "output_files/property_prediction/loss_dicts/iter_loss_dict_Egc-TESTING-GRID-highlr_20250806_024624.json",
                            "output_files/property_prediction/loss_dicts/iter_loss_dict_Egc-TESTING-GRID-highwd_20250806_170518.json",
                            "output_files/property_prediction/loss_dicts/iter_loss_dict_Egc-TESTING-GRID-lowlr_20250806_040807.json",
                            "output_files/property_prediction/loss_dicts/iter_loss_dict_Egc-TESTING-GRID-only1d_20250806_031456.json",
                            "output_files/property_prediction/loss_dicts/iter_loss_dict_Egc-TESTING-GRID-only3d_20250806_034105.json",
                            "output_files/property_prediction/loss_dicts/iter_loss_dict_Egc-TESTING-GRID-outputscale10_20250806_031502.json",
                            "output_files/property_prediction/loss_dicts/iter_loss_dict_Egc-TESTING-GRID-unfreezeencoders_20250806_040915.json",
                            "output_files/property_prediction/loss_dicts/iter_loss_dict_Egc-TESTING-GRID-unfreezeprojections_20250806_041319.json",
                            "output_files/property_prediction/loss_dicts/iter_loss_dict_Egc-TESTING-GRID-unfreezeprojections_20250806_042021.json",
                            "output_files/property_prediction/loss_dicts/iter_loss_dict_Egc-TESTING-GRID-scale12-epochs100_20250806_050013.json",
                            "output_files/property_prediction/loss_dicts/iter_loss_dict_Xc-TESTING-GRID-batchsize8_20250806_045211.json",
                            "output_files/property_prediction/loss_dicts/iter_loss_dict_Xc-TESTING-GRID-freezeencoders_20250806_035212.json",
                            "output_files/property_prediction/loss_dicts/iter_loss_dict_Xc-TESTING-GRID-freezeprojs-not-freezeencoders_20250806_043607.json",
                            "output_files/property_prediction/loss_dicts/iter_loss_dict_Xc-TESTING-GRID-highlr_20250806_025548.json",
                            "output_files/property_prediction/loss_dicts/iter_loss_dict_Xc-TESTING-GRID-highwd_20250806_035633.json",
                            "output_files/property_prediction/loss_dicts/iter_loss_dict_Xc-TESTING-GRID-highwd_20250806_171151.json",
                            "output_files/property_prediction/loss_dicts/iter_loss_dict_Xc-TESTING-GRID-lowlr_20250806_041701.json",
                            "output_files/property_prediction/loss_dicts/iter_loss_dict_Xc-TESTING-GRID-only1d_20250806_032320.json",
                            "output_files/property_prediction/loss_dicts/iter_loss_dict_Xc-TESTING-GRID-only3d_20250806_034808.json",
                            "output_files/property_prediction/loss_dicts/iter_loss_dict_Xc-TESTING-GRID-outputscale10_20250806_032414.json",
                            "output_files/property_prediction/loss_dicts/iter_loss_dict_Xc-TESTING-GRID-scale12-epochs100_20250806_051427.json",
                            "output_files/property_prediction/loss_dicts/iter_loss_dict_Xc-TESTING-GRID-unfreezeencoders_20250806_041804.json",
                            "output_files/property_prediction/loss_dicts/iter_loss_dict_Xc-TESTING-GRID-unfreezeprojections_20250806_041744.json",
                            "output_files/property_prediction/loss_dicts/iter_loss_dict_Xc-TESTING-GRID-unfreezeprojections_20250806_042338.json",
                            ]

    # finetuning loss plotting
    # for file in prediction_grid_exps_2:
    #     data_name = re.split('_|-TESTING', file)[6] # split between 'GRID'  and '_' to get variable exp part part
    #     print(data_name, file)
    #     exp_name_short = re.split('GRID|_', file)[-3] # split between 'GRID'  and '_' to get variable exp part part
    #     print('data_name, exp_name_short: ', data_name, exp_name_short)
    #     plot_prediction_logs(file_path=file,
    #         labels=('Prediction-variability', 'Train', 'Val', 'Test'),
    #         title=f"{data_name} {exp_name_short}; Standard deviation per iteration's regression predictions, together with train/valid/test losses",
    #         palette='muted',
    #         save_path=f'output_files/property_prediction/plots/plot_predictionvals_{data_name}_{exp_name_short}.png',
    #         drop_iters=True
    #         )


    finetuning_files = {"Xc":"/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_prediction_cv-cluster_Xc_split_2_20250901_051912_epoch_2000.json",
                        "Eat":"/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_prediction_cv-cluster_Eat_split_2_20250901_042458_epoch_2000.json",
                        "EPS":"/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_prediction_cv-cluster_EPS_split_2_20250901_041811_epoch_2000.json",
                        "Egb":"/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_prediction_cv-cluster_Egb_split_2_20250901_134712_epoch_2000.json",
                        "Ei":"/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_prediction_cv-cluster_Ei_split_2_20250901_152936_epoch_2000.json",
                        "Nc":"/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_prediction_cv-cluster_Nc_split_2_20250901_164156_epoch_2000.json",
                        "Eea":"/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_prediction_cv-cluster_Eea_split_2_20250901_152549_epoch_2000.json",
                        "Egc":"/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_prediction_cv-cluster_Egc_split_2_20250901_235227_epoch_1000.json",}

    pretrain_files = {"Pretraining":"/home/14044994/test/Repos/e3_diffusion_for_molecules/iter_loss_dict_final_model_combined_PI1M_no-amp_loaded-PI1M-noampfp32-encoders_20250731_051538_epoch_3.json"}
    

    # plot the prediction std together with the experiment's loss values;
    for exp_name,file in pretrain_files.items():
        for label_set in [('Train', 'Val', 'Test'),('Train_1d', 'Val_1d', 'Test_1d'), ('Train_3d', 'Val_3d', 'Test_3d'), ('Train_contrastive', 'Val_contrastive', 'Test_contrastive'),]:
            plot_prediction_logs_no_train_no_preds(file_path=file,
                                    labels=label_set,
                                    title=f"{exp_name} - train and evaluation performance on PI1M dataset",
                                    palette='muted',
                                    save_path=f'output_files/property_prediction_cv/plots/{exp_name}-performance-{label_set[0]}.png',
                                    drop_iters=True,                            
                                )

    # # plot the prediction std together with the experiment's loss values;
    # for exp_name,file in finetuning_files.items():
    #     plot_prediction_logs_no_train(file_path=file,
    #                             labels=('Prediction-variability', 'Val', 'Test', 'Test-RMSE'),
    #                             title=f"{exp_name} - downstream prediction performance alongside std of each iteration's prediction values",
    #                             palette='muted',
    #                             save_path=f'output_files/property_prediction_cv/plots/prediction-performance-{exp_name}.png',
    #                             drop_iters=True,                            
    #                         )
