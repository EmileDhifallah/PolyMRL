import math
import numpy as np
import json

import json
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def plot_mean_test_loss(
    file_paths,
    colors=None,
    title='Mean Test Loss Over Epochs',
    figsize=(10, 6),
    save_path=None,
    column_name="Test",
    jitter=True
):
    """
    Plots mean test loss values over epochs from multiple JSON files using seaborn.

    Parameters:
        file_paths (list): List of paths to JSON files containing loss dictionaries.
        colors (list): List of colors for each file. If None, uses seaborn default palette.
        title (str): Title of the plot.
        figsize (tuple): Figure size as (width, height).
        save_path (str): Path to save the plot. If None, displays the plot.
    """
    
    # Set seaborn style
    sns.set_style("whitegrid")
    plt.figure(figsize=figsize)
    
    # Default colors if not provided
    if colors is None:
        colors = sns.color_palette("husl", len(file_paths))
    
    for i, file_path in enumerate(file_paths):
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        
        # Load JSON data
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract test loss data
        if column_name not in data:
            print(f"Warning: 'Test_loss' key not found in {file_path}")
            continue
        
        test_loss_data = data[column_name]
        
        # Compute mean for each epoch
        mean_test_losses = []
        epochs = sorted(test_loss_data.keys(), key=int)
        
        for epoch in epochs:
            epoch_losses = test_loss_data[epoch]
            if isinstance(epoch_losses, list) and len(epoch_losses) > 0:
                mean_loss = np.mean(epoch_losses)
                mean_test_losses.append(mean_loss)
            else:
                print(f"Warning: Invalid data format for epoch {epoch} in {file_path}")
        
        # Create epoch indices
        epoch_indices = list(range(len(mean_test_losses)))
        
        # Extract a label from the file path (use filename without extension)
        label = Path(file_path).stem
        # Optionally, extract a shorter label from the filename
        if "iter_loss_dict_" in label:
            label = label.split("iter_loss_dict_")[1].split("split")[0]
        
        # Plot using seaborn
        if jitter:
            y_jitter=0.1
        else:
            y_jitter=0.0
        sns.lineplot(
            x=epoch_indices, 
            y=mean_test_losses, 
            label=label,
            color=colors[i] if i < len(colors) else None,
            linewidth=2,
            marker='',
            markersize=3,
            alpha=0.8,
            
        )
    
    # Customize the plot
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(f'Mean {column_name} Loss', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add grid with seaborn style
    plt.grid(True, alpha=0.3)
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    
    # Save or show plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
    # Clear the figure
    plt.clf()

def plot_mean_test_loss_2(
    file_paths,
    colors=None,
    title='Mean Loss Over Epochs',
    figsize=(10, 6),
    save_path=None,
    column_names=["Test"],  # now a list
    jitter=True,
    markers=['o']
):
    """
    Plots mean loss values over epochs from multiple JSON files using seaborn.

    Parameters:
        file_paths (list): List of paths to JSON files containing loss dictionaries.
        colors (list): List of colors for each file. If None, uses seaborn default palette.
        title (str): Title of the plot.
        figsize (tuple): Figure size as (width, height).
        save_path (str): Path to save the plot. If None, displays the plot.
        column_names (list): List of keys in the JSON files to plot.
        jitter (bool): Whether to apply slight jitter to values.
    """
    # Ensure column_names is a list
    if isinstance(column_names, str):
        column_names = [column_names]

    sns.set_style("whitegrid")
    plt.figure(figsize=figsize)

    # Default colors if not provided
    if colors is None:
        colors = sns.color_palette("husl", len(file_paths) * len(column_names))

    color_idx = 0
    for i, file_path in enumerate(file_paths):
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue

        with open(file_path, 'r') as f:
            data = json.load(f)

        # Extract label from filename
        label_base = Path(file_path).stem
        if "iter_loss_dict_" in label_base:
            label_base = label_base.split("iter_loss_dict_")[1].split("_")[0]

        for col in column_names:
            if col not in data:
                print(f"Warning: '{col}' key not found in {file_path}")
                continue

            col_data = data[col]
            mean_losses = []
            epochs = sorted(col_data.keys(), key=int)

            for epoch in epochs:
                epoch_losses = col_data[epoch]
                if isinstance(epoch_losses, list) and len(epoch_losses) > 0:
                    mean_loss = np.mean(epoch_losses)
                    mean_losses.append(mean_loss)
                else:
                    print(f"Warning: Invalid data format for epoch {epoch} in {file_path}")

            epoch_indices = list(range(len(mean_losses)))

            # Build label (file + column)
            label = f"{label_base}-{col}"

            if jitter:
                y_jitter = 0.1
            else:
                y_jitter = 0.0

            sns.lineplot(
                x=epoch_indices,
                y=mean_losses,
                label=label,
                color=colors[color_idx % len(colors)],
                linewidth=2,
                marker='',
                markersize=3,
                alpha=0.8,
            )
            color_idx += 1

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Mean Loss', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

    plt.clf()

# Example usage:
# file_list = [
#     "/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_EPS-oligomer_3001ep_20250902_161011_epoch_3000.json",
#     "/path/to/another/loss_dict.json"
# ]
# plot_mean_test_loss(file_list, title="Test Loss Comparison", save_path="test_loss_comparison.png")

def best_mean_fold_rmse(fold_dict_list):
    rmses_list = []
    rmses_std = []
    for fold_dict in fold_dict_list:
        best_rmse = math.inf
        best_rmse_std = 0.0
        for rmse_epoch in fold_dict['Test_rmses'].values():
            rmse_epoch_mean = sum(rmse_epoch) / len(rmse_epoch)
            if rmse_epoch_mean < best_rmse:
                # print('new best rmse: ', rmse_epoch_mean)
                best_rmse = rmse_epoch_mean
                best_rmse_std = np.std(rmse_epoch)
        rmses_list.append(best_rmse)
        rmses_std.append(best_rmse_std)
    # print('rmses_list: ', rmses_list)
    # print('rmses_std: ', rmses_std)
    mean_rmse_all_folds = sum(rmses_list) / len(rmses_list)
    std_rmse_all_folds = np.std(rmses_list)

    return mean_rmse_all_folds, std_rmse_all_folds


def best_mean_fold_r2(fold_dict_list):
    r2s_list = []
    r2s_std = []
    for fold_dict in fold_dict_list:
        best_r2 = -math.inf
        best_r2_std = 0.0
        for r2_epoch in fold_dict['Test_r2s'].values():
            r2_epoch_mean = sum(r2_epoch) / len(r2_epoch)
            if r2_epoch_mean > best_r2:
                # print('new best r2: ', r2_epoch_mean)
                best_r2 = r2_epoch_mean
                best_r2_std = np.std(r2_epoch)
        r2s_list.append(best_r2)
        r2s_std.append(best_r2_std)
    # print('r2s_list: ', r2s_list)
    # print('r2s_std: ', r2s_std)
    mean_r2_all_folds = sum(r2s_list) / len(r2s_list)
    std_r2_all_folds = np.std(r2s_list)

    return mean_r2_all_folds, std_r2_all_folds




if __name__ == "__main__":
    
    file_list_Eat = [
        "/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_Eat-monomers_3001ep_20250901_084812_epoch_2000.json",
        "/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_Eat-oligomer_3001ep_20250902_155158_epoch_2000.json"
    ]

    file_list_EPS = [
        "/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_EPS-monomers_3001ep_20250901_094241_epoch_3000.json",
        "/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_EPS-oligomer_3001ep_20250902_161011_epoch_3000.json"
    ]

    file_list_Xc = [
        "/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_Xc-monomers_3001ep_20250901_111749_epoch_3000.json",
        "/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_Xc-oligomer_3001ep_20250902_124001_epoch_3000.json"
    ]

    cluster_file_list_Eat = [
        "/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_prediction_cv-cluster_Eat_split_2_20250901_082702_epoch_2000.json",
        "/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_prediction_cv-no-cluster_Eat_split_2_20250901_204606_epoch_2000.json",
        "/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_prediction_cv-random_Eat_split_2_20250901_123641_epoch_2000.json"
    ]

    cluster_file_list_EPS = [
        "/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_prediction_cv-cluster_EPS_split_2_20250901_041811_epoch_2000.json",
        "/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_prediction_cv-no-cluster_EPS_split_2_20250901_204139_epoch_2000.json",
        "/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_prediction_cv-random_EPS_split_2_20250901_123216_epoch_2000.json"
    ]

    cluster_file_list_Xc = [
        "/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_prediction_cv-cluster_Xc_split_2_20250901_051912_epoch_2000.json",
        "/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_prediction_cv-no-cluster_Xc_split_2_20250901_202115_epoch_2000.json",
        "/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_prediction_cv-random_Xc_split_2_20250901_132304_epoch_2000.json"
    ]
    
    
    for col in ["Val", "Test"]:#, "Test_rmses", "Train"]:
        if col=='Train':
            jitter=True
        else:
            jitter=False
        plot_mean_test_loss(cluster_file_list_EPS, jitter=jitter, column_name=col, title="Clustered scaffold vs unclustered scaffold vs random split - Test and val loss", save_path=f"/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/plots/EPS_{col}_comparison_clusters.png")
        plot_mean_test_loss(cluster_file_list_Xc, jitter=jitter, column_name=col, title="Clustered scaffold vs unclustered scaffold vs random split - Test and val loss", save_path=f"/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/plots/Xc_{col}_comparison_clusters.png")
        plot_mean_test_loss(cluster_file_list_Eat, jitter=jitter, column_name=col, title="Clustered scaffold vs unclustered scaffold vs random split - Test and val loss", save_path=f"/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/plots/Eat_{col}_comparison_clusters.png")
    
    # column_names=['Test', 'Train']
    # plot_mean_test_loss_2(file_list_EPS, column_names=column_names, markers=['o',''], title="Oligomers - train and test loss comparison", save_path=f"/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/plots/EPS_{column_names[0]+'_'+column_names[1]}_comparison_mono-vs-oligo.png")
    # plot_mean_test_loss_2(file_list_Xc, column_names=column_names, markers=['o',''], title="Oligomers - train and test loss comparison", save_path=f"/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/plots/Xc_{column_names[0]+'_'+column_names[1]}_comparison_mono-vs-oligo.png")
    # plot_mean_test_loss_2(file_list_Eat, column_names=column_names, markers=['o',''], title="Oligomers - train and test loss comparison", save_path=f"/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/plots/Eat_{column_names[0]+'_'+column_names[1]}_comparison_mono-vs-oligo.png")



    '''
    file_dict={"XC":[
        "/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_prediction_cv-cluster_Xc_split_2_20250901_051912_epoch_2000.json",
        "/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_prediction_cv-random_Xc_split_0_20250901_093120_epoch_2000.json"
    ],
    "EPS":[
        "/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_prediction_cv-cluster_EPS_split_0_20250901_064139_epoch_5000.json",
        "/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_prediction_cv-cluster_EPS_split_2_20250901_064342_epoch_5000.json"
    ],"Eat":[
        "/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_prediction_cv-cluster_Eat_split_0_20250901_065222_epoch_5000.json",
        "/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_prediction_cv-cluster_Eat_split_2_20250901_065333_epoch_5000.json"
    ],"Egb":[
        "/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_prediction_cv-cluster_Egb_split_0_20250901_144339_epoch_3000.json",
        "/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_prediction_cv-cluster_Egb_split_2_20250901_150055_epoch_3000.json"
    ],"Eea":[
        "/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_prediction_cv-cluster_Eea_split_0_20250901_154207_epoch_4000.json",
        "/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_prediction_cv-cluster_Eea_split_2_20250901_174359_epoch_5000.json"
    ],"Ei":[
        "/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_prediction_cv-cluster_Ei_split_0_20250901_175115_epoch_5000.json",
        "/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_prediction_cv-cluster_Ei_split_2_20250901_175116_epoch_5000.json"
    ],"Nc":[
        "/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_prediction_cv-cluster_Nc_split_0_20250901_181137_epoch_4000.json",
        "/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_prediction_cv-cluster_Nc_split_2_20250901_181613_epoch_4000.json"
    ],"Egc":[
        "/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_prediction_cv-cluster_Egc_split_0_20250901_235600_epoch_1000.json",
        "/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_prediction_cv-cluster_Egc_split_2_20250901_235227_epoch_1000.json"

    ],"XC-Baseline":[
        "/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_Xc-baseline_2001ep_20250902_052015_epoch_2000.json"
    ],"EPS-Baseline":[
        "/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_EPS-baseline_2001ep_20250902_043854_epoch_2000.json"
    ],"Eat-Baseline":[
        "/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_Eat-baseline_2001ep_20250902_043856_epoch_2000.json"
    ],"Egb-Baseline":[
        "/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_Egb-baseline_2001ep_20250902_051002_epoch_2000.json"
    ],"Eea-Baseline":[
        "/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_Eea-baseline_2001ep_20250902_043641_epoch_2000.json"
    ],"Ei-Baseline":[
        "/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_Ei-baseline_2001ep_20250902_043647_epoch_2000.json"
    ],"Nc-Baseline":[
        "/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_Nc-baseline_2001ep_20250902_044110_epoch_2000.json"
    ],"Egc-Baseline":[
        "/home/14044994/test/Repos/e3_diffusion_for_molecules/output_files/property_prediction_cv/loss_dicts/iter_loss_dict_Egc-baseline_2001ep_20250902_084210_epoch_1000.json"
    ]}

    for dataset_name,files_list in file_dict.items():

        dict_list = []
        for dict_file in files_list:
            with open(dict_file) as json_file:
                data = json.load(json_file)
                dict_list.append(data)

        mean_rmse, std_rmse = best_mean_fold_rmse(dict_list)
        mean_r2, std_r2 = best_mean_fold_r2(dict_list)
        print(f"\n {dataset_name} ; results:")
        print(f"Mean RMSE across folds: {mean_rmse:.4f} ± {std_rmse:.4f}")
        print(f"Mean R² across folds:   {mean_r2:.4f} ± {std_r2:.4f}")
    '''