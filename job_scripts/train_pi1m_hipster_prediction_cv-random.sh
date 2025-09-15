#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --job-name=prediction_cv_random
#SBATCH --output=/home/14044994/test/Repos/e3_diffusion_for_molecules/script_outputs/predicting_pi1m_out_%j-cv-random.log
#SBATCH --error=/home/14044994/test/Repos/e3_diffusion_for_molecules/script_outputs/predicting_pi1m_err_%j-cv-random.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=performance
#SBATCH --gres=gpu:1

# conda activate polyMRL

## polyMRL_debug_mean_pool_comb_refl_false_diagn_contr_full_contr_test_2
## #SBATCH --time=48:00:00
## #SBATCH --nodelist=gn1,gn2,gn3

echo $CUDA_VISIBLE_DEVICES

# no coredumps
ulimit -S -c 0
ulimit -s unlimited

# print user ID and job ID
echo $USER
echo $SLURM_JOBID

## #SBATCH --gpus=2


# export WANDB_API_KEY=80faf35516859516c54d336727d50b2313505447

# no property/scaffold based training (unconditional)
# python -m train.train --run_name evaluation_self_pretrained_PI1M_weights --data_name pi1m_modified --no-only_test --num_props 0 --wandb_offline_or_online offline
# --wandb_project_name polymer_molgpt_evaluation

# one property based conditional training
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# test EGNN autoencoder
# python -m main_pi1m --n_epochs 2 --exp_name edm_qm9_data_test --n_stability_samples 1000 --wandb_usr emiledhifallah --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 1 --ema_decay 0.9999



# pretrain polyMRL on QM9 dataset
# python -m main_pi1m --datadir pi1m/temp --dataset qm9 --train_mode pretrain --n_epochs 2 --exp_name polyMRL_qm9_pretrain_hipster_2e --n_stability_samples 1000 --wandb_usr emiledhifallah --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 1 --ema_decay 0.9999
# python -m main_pi1m --datadir pi1m/temp --cls_rep_3d mean_pool --reflection_equiv True --dataset qm9 --train_mode pretrain --n_epochs 5 --exp_name QM9_mean_pool_3d_diagn_new_norm --n_stability_samples 1000 --wandb_usr emiledhifallah --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 32 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 1 --ema_decay 0.9999


# pretrain polyMRL on PI1M dataset
# python -m main_pi1m --datadir pi1m/mmpolymer_data_new/PI1M --dataset PI1M --train_mode pretrain --n_epochs 2 --exp_name polyMRL_PI1M_pretrain_hipster_2e --n_stability_samples 1000 --wandb_usr emiledhifallah --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 32 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 1 --ema_decay 0.9999
# python -m main_pi1m --datadir pi1m/mmpolymer_data_new/PI1M --cls_rep_3d mean_pool --reflection_equiv True --dataset PI1M --train_mode pretrain --n_epochs 6 --exp_name final_model_1  --n_stability_samples 1000 --wandb_usr emiledhifallah --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 32 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 1 --ema_decay 0.9999
# PI1M_mean_pool_comb_refl_false_diagn_contr_full_contr_test-2

# finetune polyMRL on Eat or other property prediction dataset
# python -m main_pi1m --datadir pi1m/mmpolymer_data_new/Eat --dataset Eat --pretrained_dir outputs/polyMRL_qm9_pretrain --train_mode property_prediction --freeze_encoders False --freeze_projections False --n_epochs 100 --exp_name polyMRL_Eat_prediction --n_stability_samples 1000 --wandb_usr emiledhifallah --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 16 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 1 --ema_decay 0.9999

# finetune polyMRL on all 8 small property prediction-datasets
data_list=("Egc")
split_list=("2") # "1" "2" "3")

for data in "${data_list[@]}"
        do
        for split in "${split_list[@]}"
                do
                        data_direc="pi1m/mm_polymer_cv_data/data_cv_random/${data}/splitting_${split}" # replace to your data path
                        data_name="${data}"
                        experi_name="${data}_split_${split}"

                        python -m main_pi1m --datadir $data_direc --dataset $data_name --mixed_precision_training False --amp_dtype 32 --lr 8e-3 --weight_decay 0.00005 --pretrained_dir outputs/final_model_combined_PI1M_no-amp_loaded-PI1M-noampfp32-encoders_epoch_3 --train_mode property_prediction --scaling_value 2.0 --only_1d False --only_3d False --variable_lrs True --unfreeze_encoders True --unfreeze_projections True --freeze_encoders True --freeze_projections True --n_epochs 2001 --batch_size 64 --exp_name prediction_cv-random_$experi_name --cls_rep_3d mean_pool --reflection_equiv True --n_stability_samples 1000 --wandb_usr emiledhifallah --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --nf 256 --n_layers 9 --normalize_factors [1,4,10] --test_epochs 1 --ema_decay 0.9999
                done
        done

# 1d pretrained checkpoint: final_model_1d_PI1M-weightdecay-no_amp-fp32_epoch_4
# 3d pretrained checkpoint: final_model_3d_PI1M-weightdecay-no_amp-fp32_epoch_6
# combined pretrained checkpoint: final_model_combined_PI1M_no-amp_loaded-PI1M-amp-encoders_epoch_3 OR final_model_combined_PI1M_no-amp_loaded-PI1M-noampfp32-encoders_epoch_3

## --weight_decay 0.00001


# #unconditional training
# python train/train.py --run_name unconditional_moses --data_name moses2 --batch_size 384 --max_epochs 10 --num_props 0

# # property based conditional training
# python train/train.py --run_name logp_moses --data_name moses2 --batch_size 384 --max_epochs 10 --props logp --num_props 1

# python train/train.py --run_name logp_sas_moses --data_name moses2 --batch_size 384 --max_epochs 10 --props logp sas --num_props 2

# # scaffold based conditional training
# python train/train.py --run_name scaffold_moses --data_name moses2 --scaffold --batch_size 384 --max_epochs 10

# # scaffold + property based conditional training
# python train/train.py --run_name logp_scaffold_moses --data_name moses2 --scaffold --batch_size 384 --num_props 1 --max_epochs 10 --props logp

# python train/train.py --run_name logp_sas_scaffold_moses --data_name moses2 --scaffold --batch_size 384 --num_props 2 --max_epochs 10 --props logp sas

# print job's end date and close job script file
MY_DATE=`date`
echo "Job finished at $MY_DATE"
exit
